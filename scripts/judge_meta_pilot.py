"""Meta judge calibration runner (issue-62 followup).

Replays the 60 pilot prompts (40 core + 20 swap, seed 62) through Meta
Model API judges and compares against the HF-router judgments from the
pilot. Source prompts are REUSED from runs/judge-pilot-62 (the Nov 9
eval logs are gone, so resampling is impossible); identity of the
render path (same templates, same seed) keeps the comparison paired.

Spend cap $5.00 HARD: `estimate` runs before `run` and aborts when the
upper-bound estimate exceeds the cap. Cost is computed from the listed
rate table (hsea/judge/providers.py), reasoning billed as output.

Commands (run from the repo root, key from env or .env):

    .venv/bin/python scripts/judge_meta_pilot.py estimate
    .venv/bin/python scripts/judge_meta_pilot.py run
    .venv/bin/python scripts/judge_meta_pilot.py analyze
    .venv/bin/python scripts/judge_meta_pilot.py plot

Reads runs/judge-pilot-62/*.jsonl read-only; writes only
``runs/judge-meta-62/`` and ``docs/reports/issue-62-meta-*.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hsea.judge.providers import (  # noqa: E402
    HF_JUDGE_A,
    HF_JUDGE_B,
    META_BASE_URL,
    META_JUDGE_CONFIGS,
    META_RATES,
    judgment_cost,
)
from hsea.judge.templates import (  # noqa: E402
    ABSTENTION_LABELS,
    ABSTENTION_SYSTEM,
    DISPATCH_SYSTEM,
    SAFETY_LABELS,
    SAFETY_SYSTEM,
    estimate_tokens,
    parse_judgment,
    parse_pairwise,
    render_abstention,
    render_dispatch_pair,
    render_safety,
)
from judge_pilot import cohen_kappa, wilson  # noqa: E402

SEED = 62
PILOT_DIR = ROOT / "runs" / "judge-pilot-62"
RUN_DIR = ROOT / "runs" / "judge-meta-62"
REPORT_DIR = ROOT / "docs" / "reports"

SPEND_CAP = 5.00
MAX_RETRIES = 3

# Which prompt sets each config judges. ember-minimal covers core only:
# it is a cost ablation, and swap-twice already has two Meta judges.
PLAN: dict[str, list[str]] = {
    "wonder-low": ["core", "swap"],
    "ember-low": ["core", "swap"],
    "ember-minimal": ["core"],
}


def _prompts(which: str) -> list[tuple[str, str, str, str]]:
    """(key, track, system, user) for core and/or swap prompts."""

    def split(full: str, system: str) -> tuple[str, str]:
        assert full.startswith(system), "render must start with system"
        return system, full[len(system):].strip()

    out: list[tuple[str, str, str, str]] = []
    if "core" in which:
        for line in open(PILOT_DIR / "inputs.jsonl"):
            rec = json.loads(line)
            if rec["track"] == "safety":
                full = render_safety(rec["input"],
                                     str(rec["metadata"].get("label")),
                                     rec["completion"], seed=SEED)
                system, user = split(full, SAFETY_SYSTEM)
            else:
                full = render_abstention(rec["input"], str(rec["metadata"].get(
                    "stakes")), str(rec["metadata"].get("label")),
                    rec["completion"], seed=SEED)
                system, user = split(full, ABSTENTION_SYSTEM)
            out.append((f"{rec['track']}:{rec['id']}:{rec['model']}",
                        rec["track"], system, user))
    if "swap" in which:
        for line in open(PILOT_DIR / "swap_inputs.jsonl"):
            pair = json.loads(line)
            pid = pair["provenance"]["record_id"]
            for order in ("orig", "swap"):
                a, b = pair["chosen"], pair["rejected"]
                if order == "swap":
                    a, b = b, a
                full = render_dispatch_pair(pair["situation"], a, b,
                                            seed=SEED)
                system, user = split(full, DISPATCH_SYSTEM)
                out.append((f"dispatch:{pid}:{order}", "dispatch",
                            system, user))
    return out


def _key() -> str:
    key = os.environ.get("META_AI_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in open(ROOT / ".env"):
            line = line.strip()
            if line.startswith("META_AI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("missing META_AI_API_KEY (env or .env)")
    return key


def cmd_estimate() -> None:
    """Upper-bound cost of the Meta calibration; aborts over the cap."""
    total_calls = 0
    upper = 0.0
    for config, sets in PLAN.items():
        model = str(META_JUDGE_CONFIGS[config]["model"])
        max_out = int(META_JUDGE_CONFIGS[config]["max_tokens"])
        rates = META_RATES[model]
        prompts = _prompts("+".join(sets))
        in_tokens = sum(estimate_tokens(s) + estimate_tokens(u)
                        for _, _, s, u in prompts)
        cost = (in_tokens * rates["input"]
                + len(prompts) * max_out * rates["output"]) / 1_000_000
        total_calls += len(prompts)
        upper += cost
        print(f"{config}: calls={len(prompts)} in~{in_tokens} "
              f"out<={len(prompts) * max_out} upper=${cost:.4f}")
    print(f"total calls={total_calls} upper-bound=${upper:.4f} "
          f"cap=${SPEND_CAP:.2f}")
    if upper > SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _call(client, model: str, effort: str, max_out: int,
          system: str, user: str) -> dict:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=max_out,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                reasoning_effort=effort)
            dt = time.time() - t0
            u = resp.usage
            det = u.completion_tokens_details
            return {
                "text": resp.choices[0].message.content or "",
                "finish": resp.choices[0].finish_reason,
                "prompt_tokens": int(u.prompt_tokens or 0),
                "completion_tokens": int(u.completion_tokens or 0),
                "reasoning_tokens": int(
                    getattr(det, "reasoning_tokens", 0) or 0),
                "latency_s": round(dt, 1),
            }
        except Exception as exc:  # noqa: BLE001, retry then record
            last = exc
            time.sleep(2.0 * (attempt + 1))
    return {"error": f"{type(last).__name__}: {str(last)[:200]}"}


def cmd_run() -> None:
    """Call each Meta config over its prompt sets; ledger every call."""
    from openai import OpenAI

    client = OpenAI(api_key=_key(), base_url=META_BASE_URL)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    for config, sets in PLAN.items():
        model = str(META_JUDGE_CONFIGS[config]["model"])
        effort = str(META_JUDGE_CONFIGS[config]["reasoning_effort"])
        max_out = int(META_JUDGE_CONFIGS[config]["max_tokens"])
        for key, track, system, user in _prompts("+".join(sets)):
            res = _call(client, model, effort, max_out, system, user)
            if "error" in res:
                row = {"key": key, "track": track, "config": config,
                       "model": model, "error": res["error"]}
            else:
                cost = judgment_cost(res["prompt_tokens"],
                                     res["completion_tokens"], model)
                row = {"key": key, "track": track, "config": config,
                       "model": model, "text": res["text"],
                       "finish": res["finish"],
                       "prompt_tokens": res["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"],
                       "reasoning_tokens": res["reasoning_tokens"],
                       "computed_cost": cost,
                       "latency_s": res["latency_s"]}
                print(f"{config} {key} in={res['prompt_tokens']} "
                      f"out={res['completion_tokens']} "
                      f"(reason={res['reasoning_tokens']}) "
                      f"${cost:.5f} {res['latency_s']}s", flush=True)
            rows.append(row)
    minutes = (time.time() - t0) / 60
    ok = [r for r in rows if "error" not in r]
    total_cost = sum(r["computed_cost"] for r in ok)
    with open(RUN_DIR / "meta_judgments.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"calls": len(ok), "errors": len(rows) - len(ok),
            "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
            "completion_tokens": sum(r["completion_tokens"] for r in ok),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in ok),
            "computed_cost": total_cost, "minutes": minutes,
            "by_config": {c: {"calls": sum(1 for r in ok
                                           if r["config"] == c),
                              "cost": sum(r["computed_cost"] for r in ok
                                          if r["config"] == c)}
                          for c in PLAN}}
    with open(RUN_DIR / "spend_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


def _load_refs() -> tuple[dict, dict, dict]:
    inputs = {}
    for line in open(PILOT_DIR / "inputs.jsonl"):
        rec = json.loads(line)
        inputs[f"{rec['track']}:{rec['id']}:{rec['model']}"] = rec
    hf: dict[str, dict[str, dict]] = {}
    for line in open(PILOT_DIR / "judgments.jsonl"):
        row = json.loads(line)
        hf.setdefault(row["key"], {})[row["judge"]] = row
    meta: dict[str, dict[str, dict]] = {}
    for line in open(RUN_DIR / "meta_judgments.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            meta.setdefault(row["key"], {})[row["config"]] = row
    return inputs, hf, meta


def cmd_analyze() -> None:
    """Parse, gold-match, Meta-vs-HF agreement, calibration, swap."""
    inputs, hf, meta = _load_refs()
    summary: dict = {"configs": {}}
    for config in PLAN:
        tracks: dict = {}
        for track in ("safety", "abstention"):
            labels = SAFETY_LABELS if track == "safety" \
                else ABSTENTION_LABELS
            keys = [k for k, r in inputs.items() if r["track"] == track]
            golds = [str(inputs[k]["metadata"].get("label")) for k in keys]
            mrows = [meta.get(k, {}).get(config, {}) for k in keys]
            parsed = [parse_judgment(r.get("text", ""), labels)
                      for r in mrows]
            n = len(keys)
            n_parse = sum(1 for p in parsed if p["ok"])
            gold_hit = sum(1 for p, g in zip(parsed, golds)
                           if p["verdict"] == g)
            cell: dict = {
                "n": n, "parse": n_parse,
                "gold_match": gold_hit,
                "gold_match_ci": wilson(gold_hit, n),
            }
            for ref_name, ref_judge in (("hf_a", HF_JUDGE_A),
                                        ("hf_b", HF_JUDGE_B)):
                ref = [parse_judgment(
                    hf.get(k, {}).get(ref_judge, {}).get("text", ""),
                    labels)["verdict"] or "none" for k in keys]
                mine = [p["verdict"] or "none" for p in parsed]
                agree = sum(1 for x, y in zip(mine, ref) if x == y)
                cell[f"agree_{ref_name}"] = agree
                cell[f"agree_{ref_name}_ci"] = wilson(agree, n)
                cell[f"kappa_{ref_name}"] = cohen_kappa(mine, ref)
            hit_scores = [p["score"] for p, g in zip(parsed, golds)
                          if p["verdict"] == g and p["score"]]
            miss_scores = [p["score"] for p, g in zip(parsed, golds)
                           if p["verdict"] != g and p["score"]]
            import statistics

            cell["calibration"] = {
                "hit_mean": round(statistics.mean(hit_scores), 2)
                if hit_scores else None,
                "hit_n": len(hit_scores),
                "miss_mean": round(statistics.mean(miss_scores), 2)
                if miss_scores else None,
                "miss_n": len(miss_scores),
            }
            rlen = [len((p["reasoning"] or "").split()) for p in parsed
                    if p["reasoning"]]
            cell["reasoning_words_mean"] = round(
                statistics.mean(rlen), 1) if rlen else None
            tracks[track] = cell
        summary["configs"][config] = tracks

    swap: dict = {}
    for config, sets in PLAN.items():
        if "swap" not in sets:
            continue
        n_swap = c_swap = 0
        for line in open(PILOT_DIR / "swap_inputs.jsonl"):
            pid = json.loads(line)["provenance"]["record_id"]
            orig = parse_pairwise(meta.get(f"dispatch:{pid}:orig", {}).get(
                config, {}).get("text", ""))
            sw = parse_pairwise(meta.get(f"dispatch:{pid}:swap", {}).get(
                config, {}).get("text", ""))
            flip = {"A": "B", "B": "A", "C": "C"}
            n_swap += 1
            if orig["verdict"] in flip and sw["verdict"] == \
                    flip[orig["verdict"]]:
                c_swap += 1
        swap[config] = {"n": n_swap, "consistent": c_swap,
                        "consistent_ci": wilson(c_swap, n_swap)}
    summary["swap"] = swap
    with open(RUN_DIR / "agreement_meta.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def cmd_plot() -> None:
    """Cost-vs-quality and Meta-vs-HF agreement figures (no API)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agreement = json.load(open(RUN_DIR / "agreement_meta.json"))
    pilot = json.load(open(PILOT_DIR / "agreement.json"))
    spend = json.load(open(RUN_DIR / "spend_meta.json"))
    pilot_spend = json.load(open(PILOT_DIR / "spend.json"))

    meta_rows = [json.loads(line) for line in
                 open(RUN_DIR / "meta_judgments.jsonl") if "error" not
                 in json.loads(line)]

    def mean_cost_1k(config: str) -> float:
        rows = [r for r in meta_rows if r["config"] == config]
        return sum(r["computed_cost"] for r in rows) / len(rows) * 1000

    def gold_rate(config: str) -> float:
        cells = agreement["configs"][config]
        hits = sum(cells[t]["gold_match"] for t in ("safety", "abstention"))
        n = sum(cells[t]["n"] for t in ("safety", "abstention"))
        return hits / n

    # Fig 1: cost per 1k judgments (log) vs core gold-match rate.
    judges = []
    for config, label in (("wonder-low", "wonder-low"),
                          ("ember-low", "ember-low"),
                          ("ember-minimal", "ember-minimal")):
        judges.append((label, mean_cost_1k(config), gold_rate(config)))
    hf_cost_1k = pilot_spend["estimated_cost"] / pilot_spend["calls"] \
        * 1000
    for name, key in (("HF v41-flash", "judge_a_gold"),
                      ("HF gemma-3-27b", "judge_b_gold")):
        hits = sum(pilot["tracks"][t][key]
                   for t in ("safety", "abstention"))
        judges.append((name, hf_cost_1k, hits / 40))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [j[0] for j in judges]
    xs = [j[1] for j in judges]
    ys = [j[2] for j in judges]
    colors = ["#c0392b", "#2980b9", "#27ae60", "#7f8c8d", "#7f8c8d"]
    ax.scatter(xs, ys, s=90, c=colors)
    offsets = {"wonder-low": (8, -16), "ember-low": (8, 6),
               "ember-minimal": (-6, -28), "HF v41-flash": (8, -26),
               "HF gemma-3-27b": (8, 6)}
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(f"{label}\n${x:.2f}/1k, {y:.2f}", (x, y),
                    textcoords="offset points",
                    xytext=offsets[label], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 judgments (log scale, measured tokens)")
    ax.set_ylabel("core gold-match rate (40 cases)")
    ax.set_ylim(0.66, 0.90)
    ax.set_title("Meta judges cost 2-137x HF Router judges at similar "
                 "gold-match")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-62-meta-cost-quality.png", dpi=150)
    plt.close(fig)

    # Fig 2: per-track Meta-vs-HF agreement with Wilson whiskers.
    tracks = ("safety", "abstention")
    x = range(len(tracks))
    width = 0.13
    fig, ax = plt.subplots(figsize=(9, 4.6))
    series = []
    for config in PLAN:
        for ref in ("hf_a", "hf_b"):
            series.append((config, ref))
    for idx, (config, ref) in enumerate(series):
        vals, lo, hi = [], [], []
        for track in tracks:
            cell = agreement["configs"][config][track]
            n = cell["n"]
            vals.append(cell[f"agree_{ref}"] / n)
            ci_lo, ci_hi = cell[f"agree_{ref}_ci"]
            lo.append(vals[-1] - ci_lo)
            hi.append(ci_hi - vals[-1])
        off = (idx - (len(series) - 1) / 2) * width
        pos = [i + off for i in x]
        ax.bar(pos, vals, width, yerr=[lo, hi], capsize=2,
               label=f"{config} vs {ref}")
    ax.set_xticks(list(x), list(tracks))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("verdict agreement (Wilson 95%)")
    ax.set_title("Meta judges agree with HF judges like HF judges "
                 "agree with each other")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-62-meta-agreement.png", dpi=150)
    plt.close(fig)
    print(f"plots -> {REPORT_DIR}/issue-62-meta-cost-quality.png "
          f"{REPORT_DIR}/issue-62-meta-agreement.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="meta judge calibration")
    parser.add_argument("command",
                        choices=["estimate", "run", "analyze", "plot"])
    args = parser.parse_args(argv)
    {"estimate": cmd_estimate, "run": cmd_run,
     "analyze": cmd_analyze, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
