"""Judge pilot runner (#62): sample, estimate, run, analyze, plot.

Pilot core: 20 safety + 20 abstention REAL logged completions, dual
judges (DeepSeek-V4.1-Flash + gemma-3-27b-it via the HF router).
Swap substudy: 10 dispatch pairs x 2 orders x 2 judges.

Spend cap $2.00 HARD: `estimate` runs before `run` and aborts when the
upper-bound estimate exceeds the cap. Every paid call records its
reported estimated_cost plus token counts for the ledger.

Commands (run from the repo root, keys from the environment)::

    .venv/bin/python scripts/judge_pilot.py sample
    .venv/bin/python scripts/judge_pilot.py estimate
    .venv/bin/python scripts/judge_pilot.py run
    .venv/bin/python scripts/judge_pilot.py analyze
    .venv/bin/python scripts/judge_pilot.py plot

Reads source .eval logs read-only (never copies them); writes only
``runs/judge-pilot-62/`` (inputs, judgments, agreement) and
``docs/reports/issue-62-*.png`` (plots).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.judge.templates import (  # noqa: E402
    ABSTENTION_LABELS,
    DISPATCH_RUBRIC,
    DISPATCH_SYSTEM,
    PAIRWISE_SCHEMA,
    SAFETY_LABELS,
    canonical_fewshots,
    estimate_tokens,
    parse_judgment,
    parse_pairwise,
    render_abstention,
    render_dispatch_pair,
    render_safety,
)

SEED = 62
RUN_DIR = ROOT / "runs" / "judge-pilot-62"
REPORT_DIR = ROOT / "docs" / "reports"

SAFETY_LOGS = {
    "muse-spark": "/tmp/expert-spork-issue-14/logs/"
                   "2026-11-09T16-35-53-00-00_safety_caDjQESKTMY37FJzB2gsz5.eval",
    "v41-flash": "/tmp/expert-spork-issue-14/logs/"
                 "2026-11-09T16-36-15-00-00_safety_kJoUXQWapFiCSQ3oEeH2qx.eval",
}
ABSTENTION_LOGS = {
    "muse-spark": "/tmp/expert-spork-issue-21/logs/"
                  "2026-11-09T16-23-47-00-00_abstention_njuR2QVGw4VUSJvXPNkLBZ.eval",
    "v41-flash": "/tmp/expert-spork-issue-21/logs/"
                 "2026-11-09T16-22-40-00-00_abstention_2FX4ZbVYAU4ED6zNZuSz7X.eval",
}

JUDGE_A = "deepseek-ai/DeepSeek-V4.1-Flash"
JUDGE_B = "google/gemma-3-27b-it"
HF_BASE_URL = "https://router.huggingface.co/v1"

# Upper-bound rates ($ per 1M tokens) for the pre-run estimate. v41 rates
# are the observed 2026-11-09 novita/baseten pair from docs/inference.md;
# gemma rates were unlisted, so the v41 pair bounds them too.
RATE_IN = 0.30
RATE_OUT = 1.20
SPEND_CAP = 2.00
OUT_TOKENS_BOUND = 600


def _read_log(path: str):
    from inspect_ai.log import read_eval_log

    return read_eval_log(path)


def _text(value) -> str:
    if isinstance(value, str):
        return value
    return str(value or "")


def _sample_records(log, source: str, model: str, track: str) -> list[dict]:
    records = []
    for s in log.samples or []:
        meta = dict(s.metadata or {})
        comp = ""
        if s.output is not None:
            comp = s.output.completion or ""
        score = list((s.scores or {}).values())[0]
        records.append({
            "source_log": source,
            "model": model,
            "track": track,
            "id": str(s.id),
            "input": _text(s.input),
            "completion": comp,
            "metadata": meta,
            "det_value": float(score.value),
            "det_answer": _text(score.answer),
            "det_metadata": {k: v for k, v in (score.metadata or {}).items()},
        })
    return records


def cmd_sample() -> None:
    """Stratified seed-62 sample: 20 safety + 20 abstention + 10 pairs."""
    rng = random.Random(SEED)
    safety: list[dict] = []
    for model, path in SAFETY_LOGS.items():
        safety.extend(_sample_records(_read_log(path), path, model,
                                      "safety"))
    abst: list[dict] = []
    for model, path in ABSTENTION_LOGS.items():
        abst.extend(_sample_records(_read_log(path), path, model,
                                    "abstention"))

    def pick(records: list[dict], n: int, label_key: str,
             force_bad: int) -> list[dict]:
        bad = [r for r in records if r["det_value"] < 1.0]
        good = [r for r in records if r["det_value"] == 1.0]
        rng.shuffle(bad)
        chosen = bad[:force_bad]
        rest = bad[force_bad:] + good
        by_cell: dict[tuple[str, str], list[dict]] = {}
        for r in rest:
            cell = (r["model"], str(r["metadata"].get(label_key)))
            by_cell.setdefault(cell, []).append(r)
        for cell in by_cell.values():
            rng.shuffle(cell)
        cells = sorted(by_cell)
        idx = 0
        while len(chosen) < n:
            cell = by_cell[cells[idx % len(cells)]]
            if cell:
                chosen.append(cell.pop())
            idx += 1
            if idx > n * len(cells) + 10:
                break
        return chosen[:n]

    safety_pick = pick(safety, 20, "label", 4)
    abst_pick = pick(abst, 20, "label", 10)

    pairs = [json.loads(line) for line in
             open(ROOT / "data" / "dispatch-pairs" / "dpo.jsonl")]
    rng.shuffle(pairs)
    pair_pick = pairs[:10]
    assert {p["priority"] for p in pair_pick} == {"P1", "P2", "P3"}, \
        "swap set must span all priorities"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "inputs.jsonl", "w") as fh:
        for rec in safety_pick + abst_pick:
            fh.write(json.dumps(rec) + "\n")
    with open(RUN_DIR / "swap_inputs.jsonl", "w") as fh:
        for pair in pair_pick:
            fh.write(json.dumps(pair) + "\n")
    print(f"safety={len(safety_pick)} abstention={len(abst_pick)} "
          f"pairs={len(pair_pick)} -> {RUN_DIR}")


def _prompts() -> list[tuple[str, str, str]]:
    """(key, system, user) for every planned judgment (no API)."""
    from hsea.judge.templates import ABSTENTION_SYSTEM, SAFETY_SYSTEM

    def split(full: str, system: str) -> tuple[str, str]:
        assert full.startswith(system), "render must start with system"
        return system, full[len(system):].strip()

    out: list[tuple[str, str, str]] = []
    for line in open(RUN_DIR / "inputs.jsonl"):
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
                    system, user))
    for line in open(RUN_DIR / "swap_inputs.jsonl"):
        pair = json.loads(line)
        pid = pair["provenance"]["record_id"]
        for order in ("orig", "swap"):
            a, b = pair["chosen"], pair["rejected"]
            if order == "swap":
                a, b = b, a
            full = render_dispatch_pair(pair["situation"], a, b, seed=SEED)
            system, user = split(full, DISPATCH_SYSTEM)
            out.append((f"dispatch:{pid}:{order}", system, user))
    assert len(out) == 40 + 20, f"planned {len(out)} prompts, want 60"
    return out


def cmd_estimate() -> None:
    """Upper-bound cost of the pilot; aborts when over the cap."""
    prompts = _prompts()
    calls = len(prompts) * 2  # dual judges
    in_tokens = sum(estimate_tokens(s) + estimate_tokens(u)
                    for _, s, u in prompts) * 2
    out_tokens = calls * OUT_TOKENS_BOUND
    upper = (in_tokens * RATE_IN + out_tokens * RATE_OUT) / 1_000_000
    print(f"prompts={len(prompts)} calls={calls} "
          f"in~{in_tokens} out<={out_tokens}")
    print(f"upper-bound=${upper:.4f} cap=${SPEND_CAP:.2f}")
    if upper > SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _client():
    from openai import OpenAI

    key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_TOKEN")
    if not key:
        raise SystemExit("missing HF_API_KEY (or HUGGINGFACE_TOKEN)")
    return OpenAI(api_key=key, base_url=HF_BASE_URL)


def cmd_run() -> None:
    """Call both judges over all prompts; ledger every call."""
    prompts = _prompts()
    client = _client()
    t0 = time.time()
    rows: list[dict] = []
    for key, system, user in prompts:
        for judge in (JUDGE_A, JUDGE_B):
            resp = client.chat.completions.create(
                model=judge, temperature=0.0, max_tokens=OUT_TOKENS_BOUND,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            msg = resp.choices[0].message.content or ""
            u = resp.usage
            cost = None
            try:
                cost = float(getattr(u, "estimated_cost", None) or 0.0)
            except (TypeError, ValueError):
                cost = None
            rows.append({
                "key": key, "judge": judge, "text": msg,
                "prompt_tokens": int(u.prompt_tokens or 0),
                "completion_tokens": int(u.completion_tokens or 0),
                "estimated_cost": cost,
            })
            print(f"{key} {judge.split('/')[-1]} "
                  f"in={u.prompt_tokens} out={u.completion_tokens} "
                  f"${(cost or 0.0):.6f}", flush=True)
    minutes = (time.time() - t0) / 60
    total_cost = sum(r["estimated_cost"] or 0.0 for r in rows)
    with open(RUN_DIR / "judgments.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"calls": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "estimated_cost": total_cost, "minutes": minutes}
    with open(RUN_DIR / "spend.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a rate (total, edge-safe)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Unweighted Cohen kappa over paired labels (None when empty)."""
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = sorted(set(a) | set(b))
    expected = sum((sum(1 for x in a if x == lab) / n)
                   * (sum(1 for y in b if y == lab) / n) for lab in labels)
    if expected >= 1.0:
        return 1.0 if agree >= 1.0 else 0.0
    return (agree - expected) / (1 - expected)


def cmd_analyze() -> None:
    """Agreement: judge-vs-deterministic, dual-judge, swap consistency."""
    inputs = {}
    for line in open(RUN_DIR / "inputs.jsonl"):
        rec = json.loads(line)
        inputs[f"{rec['track']}:{rec['id']}:{rec['model']}"] = rec
    judgments: dict[str, dict[str, dict]] = {}
    for line in open(RUN_DIR / "judgments.jsonl"):
        row = json.loads(line)
        judgments.setdefault(row["key"], {})[row["judge"]] = row

    core_rows: list[dict] = []
    for key, rec in inputs.items():
        labels = SAFETY_LABELS if rec["track"] == "safety" else \
            ABSTENTION_LABELS
        gold = str(rec["metadata"].get("label"))
        det_pred = rec["det_answer"] or None
        if rec["track"] == "abstention":
            det_pred = str(rec["det_metadata"].get("abst_pred") or "none")
        parsed = {}
        for judge in (JUDGE_A, JUDGE_B):
            row = judgments.get(key, {}).get(judge, {})
            parsed[judge] = parse_judgment(row.get("text", ""), labels)
        core_rows.append({"key": key, "track": rec["track"], "gold": gold,
                          "det_value": rec["det_value"],
                          "det_pred": det_pred,
                          "judge_a": parsed[JUDGE_A],
                          "judge_b": parsed[JUDGE_B]})

    summary: dict = {"n_core": len(core_rows), "tracks": {}}
    for track in ("safety", "abstention"):
        rows = [r for r in core_rows if r["track"] == track]
        det_ok = sum(1 for r in rows if r["det_value"] == 1.0)
        a_ok = sum(1 for r in rows
                   if r["judge_a"]["verdict"] == r["gold"])
        b_ok = sum(1 for r in rows
                   if r["judge_b"]["verdict"] == r["gold"])
        a_parse = sum(1 for r in rows if r["judge_a"]["ok"])
        b_parse = sum(1 for r in rows if r["judge_b"]["ok"])
        both = [(r["judge_a"]["verdict"] or "none",
                 r["judge_b"]["verdict"] or "none") for r in rows]
        dual_agree = sum(1 for x, y in both if x == y)
        a_det = [(r["judge_a"]["verdict"] or "none",
                  (r["det_pred"] or "none").lower()) for r in rows]
        agree_a_det = sum(1 for x, y in a_det if x == y)
        deltas = [r["key"] for r in rows
                  if (r["judge_a"]["verdict"]
                      != (r["det_pred"] or "none").lower()
                      or r["judge_b"]["verdict"]
                      != (r["det_pred"] or "none").lower())]
        summary["tracks"][track] = {
            "n": len(rows),
            "det_exact": det_ok, "det_exact_ci": wilson(det_ok, len(rows)),
            "judge_a_gold": a_ok,
            "judge_a_gold_ci": wilson(a_ok, len(rows)),
            "judge_b_gold": b_ok,
            "judge_b_gold_ci": wilson(b_ok, len(rows)),
            "judge_a_parse": a_parse, "judge_b_parse": b_parse,
            "dual_agree": dual_agree,
            "dual_agree_ci": wilson(dual_agree, len(rows)),
            "dual_kappa": cohen_kappa([x for x, _ in both],
                                      [y for _, y in both]),
            "agree_a_det": agree_a_det,
            "kappa_a_det": cohen_kappa([x for x, _ in a_det],
                                       [y for _, y in a_det]),
            "delta_keys": deltas,
        }

    swap_rows: list[dict] = []
    for line in open(RUN_DIR / "swap_inputs.jsonl"):
        pair = json.loads(line)
        pid = pair["provenance"]["record_id"]
        for judge in (JUDGE_A, JUDGE_B):
            orig = parse_pairwise(
                judgments.get(f"dispatch:{pid}:orig", {}).get(
                    judge, {}).get("text", ""))
            swap = parse_pairwise(
                judgments.get(f"dispatch:{pid}:swap", {}).get(
                    judge, {}).get("text", ""))
            swap_rows.append({"pair": pid, "judge": judge,
                              "orig": orig, "swap": swap})
    consistent = 0
    for row in swap_rows:
        o, s = row["orig"]["verdict"], row["swap"]["verdict"]
        flip = {"A": "B", "B": "A", "C": "C"}
        if o in flip and s == flip[o]:
            consistent += 1
    summary["swap"] = {"n": len(swap_rows), "consistent": consistent,
                       "consistent_ci": wilson(consistent, len(swap_rows)),
                       "conflict_rate": 1 - consistent / len(swap_rows)
                       if swap_rows else None}
    with open(RUN_DIR / "agreement.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def cmd_plot() -> None:
    """Agreement heatmaps + judge-vs-deterministic delta bars (no API)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = {}
    for line in open(RUN_DIR / "inputs.jsonl"):
        rec = json.loads(line)
        inputs[f"{rec['track']}:{rec['id']}:{rec['model']}"] = rec
    judgments: dict[str, dict[str, dict]] = {}
    for line in open(RUN_DIR / "judgments.jsonl"):
        row = json.loads(line)
        judgments.setdefault(row["key"], {})[row["judge"]] = row
    agreement = json.load(open(RUN_DIR / "agreement.json"))

    # Fig 1: per-track agreement matrices (det pred rows, judge A cols).
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.22, top=0.82,
                        wspace=0.30)
    for ax, track in zip(axes, ("safety", "abstention")):
        labels = list(SAFETY_LABELS if track == "safety"
                      else ABSTENTION_LABELS)
        grid = [[0] * len(labels) for _ in labels]
        for key, rec in inputs.items():
            if rec["track"] != track:
                continue
            det = (rec["det_answer"] or "none").lower()
            if track == "abstention":
                det = str(rec["det_metadata"].get("abst_pred")
                          or "none").lower()
            parsed = parse_judgment(
                judgments.get(key, {}).get(JUDGE_A, {}).get("text", ""),
                SAFETY_LABELS if track == "safety" else ABSTENTION_LABELS)
            judge = parsed["verdict"] or "none"
            if det in labels and judge in labels:
                grid[labels.index(det)][labels.index(judge)] += 1
        im = ax.imshow(grid, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("judge A verdict")
        ax.set_ylabel("deterministic pred")
        ax.set_title(f"{track}: det-vs-judge counts (n=20)", fontsize=11)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(grid[i][j]), ha="center", va="center",
                        fontsize=11)
    cbar_ax = fig.add_axes([0.90, 0.22, 0.025, 0.55])
    fig.colorbar(im, cax=cbar_ax, label="samples")
    fig.suptitle("Judge A agrees with deterministic preds on most rows; "
                 "off-diagonal cells are the delta audit set", fontsize=11)
    fig.savefig(REPORT_DIR / "issue-62-agreement-matrix.png", dpi=150)
    plt.close(fig)

    # Fig 2: gold-match bars with Wilson whiskers + dual agreement.
    tracks = ("safety", "abstention")
    x = range(len(tracks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series = (("det_exact", "deterministic", -1.5),
              ("judge_a_gold", "judge A (v41)", -0.5),
              ("judge_b_gold", "judge B (gemma)", 0.5),
              ("dual_agree", "dual agree", 1.5))
    for name, label, off in series:
        vals, lo, hi = [], [], []
        for track in tracks:
            cell = agreement["tracks"][track]
            n = cell["n"]
            vals.append(cell[name] / n)
            ci_lo, ci_hi = cell[name + "_ci"]
            lo.append(vals[-1] - ci_lo)
            hi.append(ci_hi - vals[-1])
        pos = [i + off * width for i in x]
        ax.bar(pos, vals, width, yerr=[lo, hi], capsize=3, label=label)
    ax.set_xticks(list(x), list(tracks))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (Wilson 95%)")
    ax.set_title("Pilot gold-match: judges vs deterministic (n=20/track)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-62-delta-bars.png", dpi=150)
    plt.close(fig)
    print(f"plots -> {REPORT_DIR}/issue-62-agreement-matrix.png "
          f"{REPORT_DIR}/issue-62-delta-bars.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="judge pilot (#62)")
    parser.add_argument("command",
                        choices=["sample", "estimate", "run", "analyze",
                                 "plot"])
    args = parser.parse_args(argv)
    {"sample": cmd_sample, "estimate": cmd_estimate, "run": cmd_run,
     "analyze": cmd_analyze, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
