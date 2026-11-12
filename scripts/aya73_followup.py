"""Issue-73 stages 3-6 follow-up: prompts, deterministic scores, XSTest, RL gate.

Builds the unified follow-up prompt file from the committed eval sets
(safety 78 + abstention 80 + redteam 30 + xstest 50 + dispatch 24 +
footage 60 = 322 rows), scores per-checkpoint generations with the
shared scorers, and decides the pre-registered GRPO-beats-RFT gate.
No GPU, no API calls: pure local recompute over committed artifacts.

Commands (run from the repo root):
    python scripts/aya73_followup.py prompts   # -> eval_prompts.jsonl
    python scripts/aya73_followup.py score     # -> det_scores.json
    python scripts/aya73_followup.py xstest    # -> xstest_scores.json
    python scripts/aya73_followup.py rewards   # -> dispatch_rewards.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.safety.data import SAFETY_SYSTEM  # noqa: E402

RUN_DIR = ROOT / "runs" / "aya73-followup"
SEED = 73
CHECKPOINTS = ("base", "sft", "dpo", "repair", "kto", "ipo", "orpo",
               "rft", "grpo")

# Generous caps on the scored tracks: greedy stops at EOS, and the
# verdict-drift analysis needs UNTRUNCATED abstention outputs.
MAX_TOKENS = {"safety": 1024, "abstention": 1024, "redteam": 1024,
              "xstest": 1024, "dispatch": 256, "footage": 128}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_prompts() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from compose_aya73 import ABSTENTION_SYSTEM  # noqa: PLC0415

    prompts: list[dict] = []
    aya73 = ROOT / "data" / "aya73"
    # eval_safety ids (bench-category-label) repeat across prompts;
    # suffix the occurrence index so every key is unique.
    seen: dict[str, int] = {}
    for rec in _read_jsonl(aya73 / "eval_safety.jsonl"):
        seen[rec["id"]] = seen.get(rec["id"], 0) + 1
        prompts.append({
            "key": f"safety:{rec['id']}:{seen[rec['id']]:02d}",
            "track": "safety",
            "system": rec["system"], "prompt": rec["prompt"],
            "gold": rec["label"], "max_tokens": MAX_TOKENS["safety"],
            "meta": {"bench": rec["bench"], "category": rec["category"],
                     "gate": rec["gate"], "trained": rec["trained"]},
        })
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for rec in _read_jsonl(ROOT / "data" / "abstention" / name):
            prompts.append({
                "key": f"abstention:{rec['id']}", "track": "abstention",
                "system": ABSTENTION_SYSTEM, "prompt": rec["prompt"],
                "gold": rec["label"],
                "max_tokens": MAX_TOKENS["abstention"],
                "meta": {"split": rec["split"], "stakes": rec["stakes"],
                         "category": rec["category"], "set": rec["set"]},
            })
    for rec in _read_jsonl(aya73 / "eval_redteam.jsonl"):
        prompts.append({
            "key": f"redteam:{rec['id']}", "track": "redteam",
            "system": rec["system"], "prompt": rec["prompt"],
            "gold": "refused",
            "max_tokens": MAX_TOKENS["redteam"],
            "meta": {"family": rec["family"], "variant": rec["variant"],
                     "severity": rec["severity"],
                     "markers": rec["markers"]},
        })
    for rec in _read_jsonl(ROOT / "data" / "xstest" / "lookalikes.jsonl"):
        prompts.append({
            "key": f"xstest:{rec['id']}", "track": "xstest",
            "system": SAFETY_SYSTEM, "prompt": rec["prompt"],
            "gold": rec["label"], "max_tokens": MAX_TOKENS["xstest"],
            "meta": {"category": rec["category"],
                     "trigger": rec.get("trigger", "")},
        })
    gold_priority = {json.loads(line)["provenance"]["record_id"]:
                     json.loads(line)["priority"]
                     for line in open(ROOT / "data" / "dispatch-pairs"
                                      / "dpo.jsonl")}
    for rec in _read_jsonl(aya73 / "eval_dispatch.jsonl"):
        prompts.append({
            "key": f"dispatch:{rec['id']}", "track": "dispatch",
            "system": rec["system"], "prompt": rec["situation"],
            "gold": gold_priority[rec["id"]],
            "max_tokens": MAX_TOKENS["dispatch"],
            "meta": {"priority": gold_priority[rec["id"]]},
        })
    for rec in _read_jsonl(aya73 / "eval_footage.jsonl"):
        prompts.append({
            "key": f"footage:{rec['id']}", "track": "footage",
            "system": "", "prompt": rec["user_text"],
            "gold": rec["gold"], "max_tokens": MAX_TOKENS["footage"],
            "meta": {"kind": rec["kind"]},
        })
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "eval_prompts.jsonl", "w") as fh:
        for rec in prompts:
            fh.write(json.dumps(rec) + "\n")
    by_track: dict[str, int] = {}
    for rec in prompts:
        by_track[rec["track"]] = by_track.get(rec["track"], 0) + 1
    print(f"prompts={len(prompts)} {by_track} "
          f"-> {RUN_DIR / 'eval_prompts.jsonl'}")


def _load_comps(ckpt: str) -> dict[str, str] | None:
    path = RUN_DIR / f"eval_{ckpt}.jsonl"
    if not path.exists():
        return None
    comps = {}
    for line in open(path):
        rec = json.loads(line)
        comps[rec["key"]] = rec.get("completion", "")
    return comps


def cmd_score() -> None:
    from hsea.safety.metrics import classify_response  # noqa: PLC0415
    from hsea.scorers.abstention import (  # noqa: PLC0415
        compute_abstention_scores)
    from hsea.scorers.footage import compute_footage  # noqa: PLC0415
    from hsea.scorers.redteam import compute_redteam_scores  # noqa: PLC0415
    from hsea.stats import wilson  # noqa: PLC0415

    prompts = _read_jsonl(RUN_DIR / "eval_prompts.jsonl")
    summary: dict = {"n_prompts": len(prompts), "checkpoints": {}}
    for ckpt in CHECKPOINTS:
        comps = _load_comps(ckpt)
        if comps is None:
            print(f"WARNING: no completions for {ckpt}, skipping")
            continue
        lanes: dict[str, list[float]] = {
            "safety": [], "gate": [], "benign": [],
            "abstention_exact": [], "abstention_soft": [],
            "abstention_eval_soft": [], "redteam_asr": [], "footage": [],
        }
        for p in prompts:
            comp = comps.get(p["key"], "")
            if p["track"] == "safety":
                hit = 1.0 if classify_response(comp) == p["gold"] else 0.0
                lanes["safety"].append(hit)
                if p["meta"]["bench"] in ("harmbench", "airbench"):
                    lanes["gate"].append(hit)
                if p["meta"]["bench"] == "benign":
                    lanes["benign"].append(hit)
            elif p["track"] == "abstention":
                v = compute_abstention_scores(comp, p["gold"])
                lanes["abstention_exact"].append(float(v["abst_exact"]))
                lanes["abstention_soft"].append(float(v["abst_soft"]))
                if p["meta"]["split"] == "eval":
                    lanes["abstention_eval_soft"].append(
                        float(v["abst_soft"]))
            elif p["track"] == "redteam":
                v = compute_redteam_scores(
                    comp, p["meta"]["family"], p["meta"]["severity"],
                    p["meta"]["markers"])
                lanes["redteam_asr"].append(float(v["asr"]))
            elif p["track"] == "footage":
                v = compute_footage(
                    comp, {"kind": p["meta"]["kind"],
                           "answer": p["gold"]})
                lanes["footage"].append(float(v["footage_composite"]))
        cell: dict = {}
        for lane, vals in lanes.items():
            n = len(vals)
            k = sum(1 for v in vals if v >= 0.5)
            lo, hi = wilson(k, n) if n else (0.0, 0.0)
            cell[lane] = {"rate": (k / n) if n else 0.0, "k": k, "n": n,
                          "lo": round(lo, 3), "hi": round(hi, 3)}
        summary["checkpoints"][ckpt] = cell
        s = cell
        print(f"{ckpt:8} safety={s['safety']['rate']:.3f} "
              f"gate={s['gate']['rate']:.3f} "
              f"abs_soft={s['abstention_soft']['rate']:.3f} "
              f"rt_asr={s['redteam_asr']['rate']:.3f} "
              f"foot={s['footage']['rate']:.3f}")
    with open(RUN_DIR / "det_scores.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {RUN_DIR / 'det_scores.json'}")


def cmd_xstest() -> None:
    from hsea.scorers.xstest import compute_xstest  # noqa: PLC0415

    prompts = [p for p in _read_jsonl(RUN_DIR / "eval_prompts.jsonl")
               if p["track"] == "xstest"]
    summary: dict = {"n_prompts": len(prompts), "checkpoints": {}}
    for ckpt in CHECKPOINTS:
        comps = _load_comps(ckpt)
        if comps is None:
            print(f"WARNING: no completions for {ckpt}, skipping")
            continue
        ok, safe_ref, safe_n, ctrl_ref, ctrl_n = 0, 0, 0, 0, 0
        for p in prompts:
            v = compute_xstest(comps.get(p["key"], ""), p["gold"])
            ok += 1 if v["xstest_correct"] == 1.0 else 0
            if p["gold"] == "comply":
                safe_n += 1
                safe_ref += 1 if v["xstest_correct"] == 0.0 else 0
            else:
                ctrl_n += 1
                ctrl_ref += 1 if v["xstest_correct"] == 1.0 else 0
        n = len(prompts)
        summary["checkpoints"][ckpt] = {
            "n": n, "exact": ok / n if n else None,
            "exact_k": ok, "over_refusal": safe_ref / safe_n if safe_n
            else None, "over_refusal_k": safe_ref, "safe_n": safe_n,
            "control_refusal": ctrl_ref / ctrl_n if ctrl_n else None,
            "control_k": ctrl_ref, "control_n": ctrl_n,
        }
        c = summary["checkpoints"][ckpt]
        print(f"{ckpt:8} exact={c['exact']:.3f} "
              f"overref={safe_ref}/{safe_n} ctrl={ctrl_ref}/{ctrl_n}")
    assert summary["checkpoints"], "no xstest completion files"
    with open(RUN_DIR / "xstest_scores.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {RUN_DIR / 'xstest_scores.json'}")


def cmd_rewards() -> None:
    from hsea.rlvr.rewards import (  # noqa: PLC0415
        passes_verifier, priority_of, rule_reward_dispatch)

    prompts = [p for p in _read_jsonl(RUN_DIR / "eval_prompts.jsonl")
               if p["track"] == "dispatch"]
    summary: dict = {"n_prompts": len(prompts), "checkpoints": {}}
    for ckpt in CHECKPOINTS:
        comps = _load_comps(ckpt)
        if comps is None:
            print(f"WARNING: no completions for {ckpt}, skipping")
            continue
        rewards, exact, no_footer, pass_v = [], 0, 0, 0
        for p in prompts:
            comp = comps.get(p["key"], "")
            gold = p["meta"]["priority"]
            rewards.append(rule_reward_dispatch(comp, gold))
            if priority_of(comp) == gold:
                exact += 1
            upper = comp.upper()
            if "DECISION:" not in upper and "ACTION:" not in upper:
                no_footer += 1
            if passes_verifier(comp, gold):
                pass_v += 1
        n = len(prompts)
        summary["checkpoints"][ckpt] = {
            "n": n, "mean_reward": sum(rewards) / n if n else None,
            "priority_exact": exact / n if n else None,
            "no_footer_rate": no_footer / n if n else None,
            "verifier_pass": pass_v / n if n else None,
        }
        c = summary["checkpoints"][ckpt]
        print(f"{ckpt:8} mean={c['mean_reward']:.3f} exact={exact}/{n} "
              f"nofooter={no_footer}/{n} pass={pass_v}/{n}")
    ckpts = summary["checkpoints"]
    if "grpo" in ckpts and "rft" in ckpts:
        grpo = ckpts["grpo"]["mean_reward"]
        rft = ckpts["rft"]["mean_reward"]
        gate = grpo > rft
        summary["grpo_beats_rft"] = gate
        print(f"GATE grpo({grpo:.3f}) > rft({rft:.3f}): "
              f"{'PASS' if gate else 'FAIL'}")
    with open(RUN_DIR / "dispatch_rewards.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {RUN_DIR / 'dispatch_rewards.json'}")


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det = json.load(open(RUN_DIR / "det_scores.json"))["checkpoints"]
    rew = json.load(open(RUN_DIR / "dispatch_rewards.json"))["checkpoints"]
    ckpts = [c for c in CHECKPOINTS if c in det]
    lanes = [("safety", "safety"), ("gate", "gate"),
             ("abstention_soft", "abst soft"), ("redteam_asr", "RT ASR")]
    x = range(len(lanes))
    width = min(0.1, 0.8 / max(1, len(ckpts)))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, ckpt in enumerate(ckpts):
        vals = [det[ckpt][lane]["rate"] for lane, _ in lanes]
        pos = [j + (i - (len(ckpts) - 1) / 2) * width for j in x]
        ax.bar(pos, vals, width, label=ckpt)
    ax.set_xticks(list(x), [label for _, label in lanes])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate (RT ASR lower better, rest higher)")
    ax.set_title("Deterministic lanes per checkpoint (#73 follow-up)")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "reports" / "issue-73-lanes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    means = [rew[c]["mean_reward"] for c in ckpts]
    exact = [rew[c]["priority_exact"] for c in ckpts]
    x = range(len(ckpts))
    ax.bar([i - 0.2 for i in x], means, 0.4, label="mean rule reward")
    ax.bar([i + 0.2 for i in x], exact, 0.4, label="priority exact")
    ax.set_xticks(list(x), ckpts, rotation=15, ha="right")
    ax.set_ylim(0, 1.3)
    ax.set_title("Dispatch rewards per checkpoint (#73, held-out 24)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "reports" / "issue-73-dispatch-arms.png",
                dpi=150)
    plt.close(fig)
    print("plots -> docs/reports/issue-73-{lanes,dispatch-arms}.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="issue-73 follow-up")
    parser.add_argument("command",
                        choices=["prompts", "score", "xstest", "rewards",
                                 "plot"])
    args = parser.parse_args(argv)
    {"prompts": cmd_prompts, "score": cmd_score, "xstest": cmd_xstest,
     "rewards": cmd_rewards, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
