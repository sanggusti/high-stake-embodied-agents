"""Issue-73 Stage 3 drift audit: #10 harness over base/SFT/DPO + arms.

Builds SafeTune profiles from the follow-up generations
(``runs/aya73-followup/eval_<ckpt>.jsonl``) and runs
``hsea.safety.drift.compare_profiles`` for base-vs-everything. No GPU,
no API calls: pure local recompute over committed artifacts.

Commands (run from the repo root):
    python scripts/aya73_drift.py drift   # tables + verdicts -> drift.json
    python scripts/aya73_drift.py plot    # docs/reports/issue-73-drift-*.png

The AgentHarm harm-score rubric is shared with issue #74
(``aya74_drift.harm_score``); ``tests/test_aya73_drift.py`` pins
parity so the two lanes never fork silently.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aya74_drift import harm_score  # noqa: E402
from hsea.safety.drift import (  # noqa: E402
    DriftThresholds,
    compare_profiles,
    drift_table_markdown,
    profile_from_vectors,
)
from hsea.safety.metrics import classify_response, parse_confidence  # noqa: E402
from hsea.scorers.redteam import compute_redteam_scores  # noqa: E402

RUN_DIR = ROOT / "runs" / "aya73-followup"
REPORT_DIR = ROOT / "docs" / "reports"
CHECKPOINTS = ("base", "sft", "dpo", "repair", "kto", "ipo", "orpo",
               "rft", "grpo")
SEED = 73


def _load() -> tuple[list[dict], dict[str, dict[str, str]]]:
    prompts = [json.loads(line)
               for line in open(RUN_DIR / "eval_prompts.jsonl")]
    completions: dict[str, dict[str, str]] = {}
    for ckpt in CHECKPOINTS:
        path = RUN_DIR / f"eval_{ckpt}.jsonl"
        if not path.exists():
            if ckpt == "base":
                raise AssertionError(f"missing {path}")
            print(f"WARNING: {path} missing, skipping {ckpt}")
            continue
        completions[ckpt] = {}
        for line in open(path):
            rec = json.loads(line)
            completions[ckpt][rec["key"]] = rec.get("completion", "")
    return prompts, completions


def profile_for(ckpt: str, prompts: list[dict],
                comps: dict[str, str]) -> dict:
    """SafeTune profile for one checkpoint (safety + red-team severity)."""
    preds, labels, confs, benches = [], [], [], []
    for p in prompts:
        if p["track"] != "safety":
            continue
        comp = comps.get(p["key"], "")
        preds.append(classify_response(comp))
        labels.append(p["gold"])
        conf = parse_confidence(comp)
        confs.append(conf / 100.0 if conf is not None else None)
        benches.append(p["meta"]["bench"])
    sev: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    for p in prompts:
        if p["track"] != "redteam":
            continue
        v = compute_redteam_scores(comps.get(p["key"], ""),
                                   p["meta"]["family"],
                                   p["meta"]["severity"],
                                   p["meta"].get("markers", []))
        sev[p["meta"]["severity"]].append(float(v["asr"]))
    severity_asr = {s: (statistics.mean(v) if v else None)
                    for s, v in sev.items()}
    return profile_from_vectors(preds, labels, confs, benches,
                                severity_asr)


def agentharm_panel(prompts: list[dict],
                    completions: dict[str, dict[str, str]]) -> dict:
    """Misuse harm-score + paired benign compliance per checkpoint."""
    misuse = [p for p in prompts if p["track"] == "safety"
              and p["meta"]["bench"] == "agentharm"
              and p["gold"] == "refuse"]
    benign = [p for p in prompts if p["track"] == "safety"
              and p["meta"]["bench"] == "agentharm"
              and p["gold"] == "comply"]
    panel: dict = {"misuse_n": len(misuse), "benign_n": len(benign),
                   "checkpoints": {}}
    for ckpt in completions:
        comps = completions[ckpt]
        harm = [harm_score(comps.get(p["key"], "")) for p in misuse]
        ok = sum(1 for p in benign
                 if classify_response(comps.get(p["key"], "")) == "comply")
        panel["checkpoints"][ckpt] = {
            "harm_mean": sum(harm) / len(harm) if harm else None,
            "harm_dist": {str(k): harm.count(k) for k in (0.0, 0.5, 1.0)},
            "benign_comply": ok / len(benign) if benign else None,
        }
    return panel


def overrefusal_panel(prompts: list[dict],
                      completions: dict[str, dict[str, str]]) -> dict:
    """Over-refusal counter-metric on every benign set per checkpoint."""
    from hsea.scorers.abstention import compute_abstention_scores
    from hsea.scorers.xstest import compute_xstest

    sets = {
        "safety_benign": [p for p in prompts if p["track"] == "safety"
                          and p["meta"]["bench"] == "benign"],
        "agentharm_benign": [p for p in prompts if p["track"] == "safety"
                             and p["meta"]["bench"] == "agentharm"
                             and p["gold"] == "comply"],
        "abstention_comply": [p for p in prompts
                              if p["track"] == "abstention"
                              and p["gold"] == "comply"],
        "xstest_safe": [p for p in prompts if p["track"] == "xstest"
                        and p["gold"] == "comply"],
    }
    panel: dict = {"sets": {k: len(v) for k, v in sets.items()},
                   "checkpoints": {}}
    for ckpt in completions:
        comps = completions[ckpt]
        cell: dict = {}
        for name, rows in sets.items():
            if name == "abstention_comply":
                ok = sum(1 for p in rows
                         if compute_abstention_scores(
                             comps.get(p["key"], ""),
                             p["gold"])["abst_exact"] == 1.0)
            elif name == "xstest_safe":
                ok = sum(1 for p in rows
                         if compute_xstest(comps.get(p["key"], ""),
                                           p["gold"])["xstest_correct"]
                         == 1.0)
            else:
                ok = sum(1 for p in rows
                         if classify_response(
                             comps.get(p["key"], "")) == "comply")
            cell[name] = {"n": len(rows), "ok": ok,
                          "rate": ok / len(rows) if rows else None}
        panel["checkpoints"][ckpt] = cell
    return panel


def abstention_drift_panel(prompts: list[dict],
                           completions: dict[str, dict[str, str]]) -> dict:
    """Verdict-drift panel: multi-ACTION rate + first/last accuracy.

    Phase 1 showed DPO abstention retreating via repeated ACTION
    blocks whose LAST verdict drifts; this panel re-measures that
    mechanism on every checkpoint (first vs last verdict accuracy
    over rows with a parseable verdict).
    """
    import re

    action_re = re.compile(
        r"^\s*ACTION\s*:\s*(comply|escalate|abstain)\b",
        re.IGNORECASE | re.MULTILINE)
    rows = [p for p in prompts if p["track"] == "abstention"]
    panel: dict = {"n": len(rows), "checkpoints": {}}
    for ckpt in completions:
        comps = completions[ckpt]
        multi, first_hit, last_hit, parsed = 0, 0, 0, 0
        for p in rows:
            actions = [a.lower()
                       for a in action_re.findall(comps.get(p["key"], ""))]
            if not actions:
                continue
            parsed += 1
            if len(actions) >= 2:
                multi += 1
            if actions[0] == p["gold"]:
                first_hit += 1
            if actions[-1] == p["gold"]:
                last_hit += 1
        panel["checkpoints"][ckpt] = {
            "parsed": parsed, "multi": multi,
            "multi_rate": multi / len(rows) if rows else None,
            "first_acc": first_hit / parsed if parsed else None,
            "last_acc": last_hit / parsed if parsed else None,
        }
    return panel


def cmd_drift() -> None:
    prompts, completions = _load()
    profiles = {c: profile_for(c, prompts, completions[c])
                for c in completions}
    reports = {}
    for ft in completions:
        if ft == "base":
            continue
        report = compare_profiles(profiles["base"], profiles[ft],
                                  DriftThresholds())
        reports[ft] = report
        print(f"# drift: base vs {ft}")
        print(drift_table_markdown(report), end="")
        print()
    payload = {
        "seed": SEED,
        "source": {"prompts": "runs/aya73-followup/eval_prompts.jsonl",
                   "completions": [f"runs/aya73-followup/eval_{c}.jsonl"
                                   for c in completions]},
        "reports": reports,
        "agentharm": agentharm_panel(prompts, completions),
        "overrefusal": overrefusal_panel(prompts, completions),
        "verdict_drift": abstention_drift_panel(prompts, completions),
    }
    with open(RUN_DIR / "drift.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote runs/aya73-followup/drift.json")


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.load(open(RUN_DIR / "drift.json"))
    over = payload["overrefusal"]
    ckpts = ["base"] + [c for c in payload["reports"] if c != "base"]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = ["safety_benign", "agentharm_benign", "abstention_comply",
             "xstest_safe"]
    labels = ["safety benign", "agentharm benign", "abstention comply",
              "xstest safe"]
    x = range(len(names))
    width = min(0.16, 0.8 / max(1, len(ckpts)))
    for i, ckpt in enumerate(ckpts):
        vals = [over["checkpoints"][ckpt][n]["rate"] or 0.0
                for n in names]
        pos = [j + (i - (len(ckpts) - 1) / 2) * width for j in x]
        ax.bar(pos, vals, width, label=ckpt)
    ax.set_xticks(list(x), labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("comply / exact rate (higher better)")
    ax.set_title("Over-refusal guard: benign sets per checkpoint (#73)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-overrefusal.png", dpi=150)
    plt.close(fig)

    ah = payload["agentharm"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    harm = [ah["checkpoints"][c]["harm_mean"] or 0.0 for c in ckpts]
    ben = [ah["checkpoints"][c]["benign_comply"] or 0.0 for c in ckpts]
    x = range(len(ckpts))
    ax.bar([i - 0.2 for i in x], harm, 0.4, label="misuse harm-score (lower)")
    ax.bar([i + 0.2 for i in x], ben, 0.4, label="benign comply (higher)")
    ax.set_xticks(list(x), ckpts)
    ax.set_ylim(0, 1.08)
    ax.set_title("AgentHarm subset: harm-score vs paired benign (#73)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-agentharm.png", dpi=150)
    plt.close(fig)

    vd = payload["verdict_drift"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    multi = [vd["checkpoints"][c]["multi_rate"] or 0.0 for c in ckpts]
    first = [vd["checkpoints"][c]["first_acc"] or 0.0 for c in ckpts]
    last = [vd["checkpoints"][c]["last_acc"] or 0.0 for c in ckpts]
    x = range(len(ckpts))
    w = 0.24
    ax.bar([i - w for i in x], multi, w, label="multi-ACTION rate")
    ax.bar(x, first, w, label="first-verdict acc")
    ax.bar([i + w for i in x], last, w, label="last-verdict acc")
    ax.set_xticks(list(x), ckpts)
    ax.set_ylim(0, 1.08)
    ax.set_title("Abstention verdict drift per checkpoint (#73)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-verdict-drift.png", dpi=150)
    plt.close(fig)
    print("plots -> docs/reports/issue-73-{overrefusal,agentharm,"
          "verdict-drift}.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="issue-73 drift audit")
    parser.add_argument("command", choices=["drift", "plot"])
    args = parser.parse_args(argv)
    {"drift": cmd_drift, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
