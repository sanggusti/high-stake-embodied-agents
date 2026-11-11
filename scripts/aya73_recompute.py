"""Before/after recompute for issue-73 (shared model-comparison protocol).

Reads the aya73 inspect .eval logs plus served footage completions and
emits the canonical before/after table with columns
(metric, base, finetune, delta), finetune = the DPO checkpoint, with
Wilson 95% CIs and McNemar p for base-vs-finetune. Reuses the
scripts/recompute_metrics.py loader (load_samples) and hsea.stats, so
the numbers trace to the same machinery as #61.

Usage: python3 scripts/aya73_recompute.py [--logs-dir logs]
Writes runs/aya73-recompute/before_after.json; prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from recompute_metrics import load_samples  # noqa: E402

from hsea.stats import mcnemar, wilson  # noqa: E402

CKPTS = ("tinyaya73-base", "tinyaya73-sft", "tinyaya73-dpo")
TASKS = {"safety": "safety_score", "abstention": "abstention_score",
         "redteam": "redteam_asr"}
RUN_DIR = ROOT / "runs" / "aya73-recompute"


def _find_logs(logs_dir: Path) -> dict[tuple[str, str], Path]:
    from inspect_ai.log import read_eval_log

    found: dict[tuple[str, str], Path] = {}
    for path in sorted(logs_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(path))
        except Exception:  # noqa: BLE001, skip unreadable logs
            continue
        task = getattr(log.eval, "task", "") or ""
        model = getattr(log.eval, "model", "") or ""
        for ckpt in CKPTS:
            if ckpt in model and task in TASKS:
                found[(task, ckpt)] = path
    return found


def _rate(vals: list[float]) -> dict:
    n = len(vals)
    k = sum(1 for v in vals if v >= 0.5)
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    return {"rate": (k / n) if n else 0.0, "k": k, "n": n,
            "lo": round(lo, 3), "hi": round(hi, 3)}


def _delta(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    ab = [1.0 if x >= 0.5 else 0.0 for x in a[:n]]
    bb = [1.0 if x >= 0.5 else 0.0 for x in b[:n]]
    stat = mcnemar(ab, bb)
    return {"delta": round(sum(bb) / n - sum(ab) / n, 4) if n else 0.0,
            "p": round(float(stat.get("p_value", 1.0)), 4), "n": n}


def _values(rows: list[dict], key: str) -> list[float]:
    return [float(r["scores"].get(key, 0.0) or 0.0) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()
    logs_dir = ROOT / args.logs_dir
    found = _find_logs(logs_dir)
    print("logs:", {f"{t}/{c}": p.name for (t, c), p in found.items()})

    data: dict[str, dict[str, list[dict]]] = {}
    for task, scorer in TASKS.items():
        for ckpt in CKPTS:
            path = found.get((task, ckpt))
            if path is None:
                print(f"WARN no log for {task}/{ckpt}")
                continue
            data.setdefault(task, {})[ckpt] = load_samples(path, scorer)

    # Metric definitions: (label, task, score key, row filter).
    metrics = [
        ("safety label-match (full 78)", "safety", "safety_correct",
         lambda r: True),
        ("refusal gate: harmbench+airbench", "safety", "safety_correct",
         lambda r: r["metadata"].get("bench") in ("harmbench", "airbench")),
        ("benign comply", "safety", "safety_correct",
         lambda r: r["metadata"].get("bench") == "benign"),
        ("abstention soft accuracy", "abstention", "abst_soft",
         lambda r: True),
        ("abstention eval-split soft accuracy", "abstention", "abst_soft",
         lambda r: r["metadata"].get("split") == "eval"),
        ("redteam ASR (lower better)", "redteam", "asr",
         lambda r: True),
    ]

    table = []
    for label, task, key, filt in metrics:
        per = {}
        for ckpt in CKPTS:
            rows = [r for r in data.get(task, {}).get(ckpt, [])
                    if filt(r)]
            per[ckpt] = _values(rows, key)
        base, sft, dpo = per[CKPTS[0]], per[CKPTS[1]], per[CKPTS[2]]
        row = {"metric": label,
               "base": _rate(base), "sft": _rate(sft),
               "finetune": _rate(dpo),
               "delta": _delta(base, dpo)}
        table.append(row)

    table.extend(_footage_rows())

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "before_after.json", "w") as fh:
        json.dump(table, fh, indent=2)

    print("")
    print("| metric | base | finetune | delta |")
    print("|---|---|---|---|")
    for row in table:
        b, f, d = row["base"], row["finetune"], row["delta"]
        print(f"| {row['metric']} | {b['rate']:.3f} ({b['k']}/{b['n']}) "
              f"| {f['rate']:.3f} ({f['k']}/{f['n']}) "
              f"| {d['delta']:+.3f} (p={d['p']}) |")
    print("")
    print("| metric | base CI | sft | finetune CI |")
    print("|---|---|---|---|")
    for row in table:
        b, s, f = row["base"], row["sft"], row["finetune"]
        print(f"| {row['metric']} | {b['lo']:.3f}-{b['hi']:.3f} "
              f"| {s['rate']:.3f} ({s['k']}/{s['n']}) "
              f"| {f['lo']:.3f}-{f['hi']:.3f} |")


def _footage_rows() -> list[dict]:
    from hsea.scorers.footage import compute_footage

    serve = ROOT / "runs" / "aya73-serve"
    probes = [json.loads(line) for line in
              open(ROOT / "data" / "aya73" / "eval_footage.jsonl")]
    by_id = {p["id"]: p for p in probes}
    per: dict[str, list[float]] = {}
    for ckpt in CKPTS:
        path = serve / f"{ckpt}_footage.jsonl"
        if not path.exists():
            print(f"WARN no served footage for {ckpt}")
            per[ckpt] = []
            continue
        vals = []
        for line in open(path):
            rec = json.loads(line)
            probe = by_id[rec["id"]]
            vals.append(float(compute_footage(
                rec.get("completion", ""),
                {"kind": probe["kind"],
                 "answer": probe["gold"]})["footage_composite"]))
        per[ckpt] = vals
    base, sft, dpo = per[CKPTS[0]], per[CKPTS[1]], per[CKPTS[2]]
    return [{"metric": "footage probe composite (text-only, 60)",
             "base": _rate(base), "sft": _rate(sft),
             "finetune": _rate(dpo), "delta": _delta(base, dpo)}]


if __name__ == "__main__":
    main()
