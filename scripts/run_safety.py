"""One-command safety harness (#14): full stack vs >=1 models, logged.

Default matrix (no --model): Baseten muse-spark-open-1026 + HF
DeepSeek-V4.1-Flash. Each run is recorded via ``record_run`` (scorer
``safety_score``) and a per-bench summary (rates + ECE + risk-coverage)
is written to ``runs/<run-id>/safety_summary.json``.

Example:
  python scripts/run_safety.py --model openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hsea.aggregate import record_run  # noqa: E402
from hsea.models import baseten_model_id, hf_model_id  # noqa: E402
from hsea.safety.data import BENCHES  # noqa: E402
from hsea.safety.metrics import (  # noqa: E402
    abstention_rate,
    benign_compliance,
    decision_correct,
    escalation_recall,
    expected_calibration_error,
    refusal_rate,
    risk_coverage_curve,
)

DEFAULT_MATRIX = [
    baseten_model_id("meta-models/muse-spark-open-1026"),
    hf_model_id("deepseek-ai/DeepSeek-V4.1-Flash"),
]

SCORER_NAME = "safety_score"


def slug(model: str) -> str:
    return model.split("/")[-1].lower().replace("_", "-")[:24]


def summarize(log_path: str) -> dict:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    log = read_eval_log(log_path)
    per_bench: dict[str, dict[str, list]] = {}
    for sample in log.samples or []:
        meta = dict(sample.metadata or {})
        bench = str(meta.get("bench", "?"))
        label = str(meta.get("label", "comply"))
        score = (sample.scores or {}).get(SCORER_NAME)
        pred = str(getattr(score, "answer", "comply") or "comply")
        smeta = dict(getattr(score, "metadata", None) or {})
        conf = smeta.get("confidence")
        bucket = per_bench.setdefault(bench, {"pred": [], "label": [],
                                              "pairs": []})
        bucket["pred"].append(pred)
        bucket["label"].append(label)
        bucket["pairs"].append((conf if isinstance(conf, (int, float)) else None,
                               decision_correct(pred, label)))
    summary = {"benches": {}, "overall": {}}
    all_pairs = [(c, v) for b in per_bench.values() for c, v in b["pairs"]
                 if c is not None]
    for bench, b in sorted(per_bench.items()):
        pairs = [(c, v) for c, v in b["pairs"] if c is not None]
        summary["benches"][bench] = {
            "n": len(b["pred"]),
            "correct": sum(decision_correct(p, lab) for p, lab
                           in zip(b["pred"], b["label"])) / len(b["pred"]),
            "refusal_rate": refusal_rate(b["pred"], b["label"]),
            "escalation_recall": escalation_recall(b["pred"], b["label"]),
            "abstention_rate": abstention_rate(b["pred"], b["label"]),
            "benign_compliance": benign_compliance(b["pred"], b["label"]),
            "ece": expected_calibration_error(pairs),
            "risk_coverage": risk_coverage_curve(pairs),
        }
    summary["overall"] = {
        "n": sum(len(b["pred"]) for b in per_bench.values()),
        "ece": expected_calibration_error(all_pairs),
        "risk_coverage": risk_coverage_curve(all_pairs),
    }
    return summary


def run_one(model: str, bench: str, limit: int | None, runs_dir: str,
            log_dir: str, tag: str) -> dict:
    from inspect_ai import eval as inspect_eval  # noqa: PLC0415

    os.environ["HSEA_SAFETY_BENCH"] = bench
    if limit:
        os.environ["HSEA_SAFETY_LIMIT"] = str(limit)
    logs = inspect_eval("evals/safety.py", model=model, log_dir=log_dir,
                        limit=None)
    log_path = str(logs[0].location)
    run_id = f"{tag}-{slug(model)}"
    record_run(run_id, task="safety", model=model,
               scenario=f"safety-{bench}", log_path=log_path,
               runs_dir=runs_dir, notes=[f"bench={bench}"],
               scorer_name=SCORER_NAME)
    summary = summarize(log_path)
    summary.update({"run_id": run_id, "model": model, "bench": bench,
                    "log": log_path})
    out = Path(runs_dir) / run_id / "safety_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary


def table_row(summary: dict) -> list[str]:
    rows = []
    for bench in BENCHES:
        b = summary["benches"].get(bench)
        if not b:
            continue
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        rows.append("| " + " | ".join([
            slug(summary["model"]), bench, str(b["n"]), fmt(b["correct"]),
            fmt(b["refusal_rate"]), fmt(b["escalation_recall"]),
            fmt(b["abstention_rate"]), fmt(b["benign_compliance"]),
            fmt(b["ece"]),
        ]) + " |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the safety stack.")
    parser.add_argument("--model", action="append", default=[],
                        help="Model id (repeatable; default: 2-model matrix).")
    parser.add_argument("--bench", default="all")
    parser.add_argument("--limit", type=int, default=0,
                        help="Per-bench item cap (0 = all).")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--tag", default="safety")
    args = parser.parse_args()
    models = args.model or DEFAULT_MATRIX
    print("| model | bench | n | correct | refusal | escal | abstain | "
          "benign | ece |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model in models:
        summary = run_one(model, args.bench, args.limit or None,
                          args.runs, args.log_dir, args.tag)
        for row in table_row(summary):
            print(row, flush=True)
        print(f"# {summary['run_id']}: {summary['log']}", flush=True)


if __name__ == "__main__":
    main()
