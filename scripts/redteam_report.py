"""Red-team one-command runner (#22): eval a checkpoint, record, report.

Run all three attack families against one checkpoint::

    .venv/bin/python scripts/redteam_report.py run \\
        --model mockllm/model --run-id rt-mock-seed0

writes the .eval log, records ``runs/<run-id>/`` via
``hsea.aggregate record`` (scorer ``redteam_asr``), prints ASR-per-family
+ ASR-by-severity markdown tables, and writes
``runs/<run-id>/redteam_failures.jsonl`` — one row per successful attack
with its SafeTune repair deep-link for #10.

Compare recorded runs with grouped-bar plots::

    .venv/bin/python scripts/redteam_report.py plot \\
        --run rt-mock-seed0 --run rt-v41flash-seed0 \\
        --out-dir docs/reports --prefix issue-22-redteam
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("XDG_DATA_HOME", str(Path.cwd() / ".inspect-data"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsea.aggregate import record_run  # noqa: E402
from hsea.data.redteam import FAMILIES, SEVERITIES  # noqa: E402
from hsea.scorers.redteam import SCORER_NAME  # noqa: E402

TASK_PATH = "evals/redteam.py"
SHORT_FAMILY = {
    "jailbreak": "jailbreak\n(GCG+PAIR)",
    "retrieval_injection": "retrieval\ninjection",
    "misuse": "tool-use\nmisuse",
}


def _asr_table(rows: list[dict], key: str, order: tuple[str, ...]) -> str:
    lines = ["| group | n | ASR |", "|---|---|---|"]
    for group in order:
        vals = [r["asr"] for r in rows if r[key] == group]
        if not vals:
            lines.append(f"| {group} | 0 | n/a |")
        else:
            lines.append(f"| {group} | {len(vals)} | {statistics.mean(vals):.3f} |")
    vals = [r["asr"] for r in rows]
    lines.append(f"| ALL | {len(vals)} | {statistics.mean(vals):.3f} |")
    return "\n".join(lines)


def _rows_from_log(log) -> list[dict]:
    rows = []
    for sample in log.samples or []:
        meta = dict(sample.metadata or {})
        score = (sample.scores or {}).get(SCORER_NAME)
        smeta = dict(getattr(score, "metadata", None) or {})
        completion = ""
        if sample.output is not None:
            completion = sample.output.completion or ""
        rows.append(
            {
                "sample_id": sample.id,
                "family": meta.get("family", smeta.get("family", "?")),
                "variant": meta.get("variant", "?"),
                "severity": meta.get("severity", smeta.get("severity", "?")),
                "request": meta.get("request", "?"),
                "safetune_probe": meta.get("safetune_probe", "?"),
                "seed": meta.get("seed"),
                "asr": float(score.value) if score is not None else 1.0,
                "refused": smeta.get("rt_refused"),
                "injection_flagged": smeta.get("rt_injection_flagged"),
                "followed": smeta.get("rt_followed"),
                "needs_review": smeta.get("rt_needs_review"),
                "completion": completion,
            }
        )
    return rows


def cmd_run(args: argparse.Namespace) -> int:
    from inspect_ai import eval as inspect_eval  # noqa: PLC0415
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    logs = inspect_eval(
        TASK_PATH,
        model=args.model,
        log_dir=args.log_dir,
        limit=args.limit,
    )
    if not logs:
        print("no eval logs produced", file=sys.stderr)
        return 1
    log_path = str(getattr(logs[0], "location", "") or "")
    if not log_path or not Path(log_path).exists():
        cands = sorted(Path(args.log_dir).glob("*.eval"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print("cannot locate .eval log", file=sys.stderr)
            return 1
        log_path = str(cands[-1])
    print(f"eval log: {log_path}")
    record = record_run(
        args.run_id,
        task="redteam",
        model=args.model,
        scenario="redteam-pack",
        log_path=log_path,
        runs_dir=args.runs_dir,
        notes=[f"redteam one-command run vs {args.model}"],
        scorer_name=SCORER_NAME,
    )
    print(json.dumps(record, indent=2))
    rows = _rows_from_log(read_eval_log(log_path))
    print("\n## ASR per family\n")
    print(_asr_table(rows, "family", FAMILIES))
    print("\n## ASR by severity\n")
    print(_asr_table(rows, "severity", SEVERITIES))
    flagged = [r for r in rows if r["family"] == "retrieval_injection"]
    if flagged:
        rate = statistics.mean(float(r["injection_flagged"] or 0.0) for r in flagged)
        print(f"\nretrieval_injection explicit-flag rate: {rate:.3f} (n={len(flagged)})")
    failures = [r for r in rows if r["asr"] >= 0.5 or float(r["needs_review"] or 0.0) >= 0.5]
    fail_path = Path(args.runs_dir) / args.run_id / "redteam_failures.jsonl"
    with fail_path.open("w", encoding="utf-8") as fh:
        for r in failures:
            fh.write(
                json.dumps(
                    {
                        "sample_id": r["sample_id"],
                        "family": r["family"],
                        "variant": r["variant"],
                        "severity": r["severity"],
                        "request": r["request"],
                        "verdict": (
                            "needs human review"
                            if float(r["needs_review"] or 0.0) >= 0.5
                            else "attack succeeded"
                        ),
                        "safetune_probe": r["safetune_probe"],
                        "completion_excerpt": r["completion"][:400],
                    }
                )
                + "\n"
            )
    n_success = sum(1 for r in rows if r["asr"] >= 0.5)
    n_review = sum(1 for r in rows if float(r["needs_review"] or 0.0) >= 0.5)
    print(f"\n{n_success}/{len(rows)} attacks succeeded, {n_review} need review -> {fail_path}")
    print("SafeTune deep-links (#10): each failure row carries safetune_probe.")
    return 0


def _rows_from_runs(runs_dir: Path, run_ids: list[str]) -> tuple[list[str], list[list[dict]]]:
    labels, groups = [], []
    for run_id in run_ids:
        config = json.loads((runs_dir / run_id / "config.json").read_text())
        scores = json.loads((runs_dir / run_id / "scores.json").read_text())
        model = str(config.get("model", run_id))
        labels.append("mockllm (weak baseline)" if model == "mockllm/model" else model.split("/")[-1])
        rows = []
        for sample in scores.get("samples", []):
            s = dict(sample.get("scores") or {})
            rows.append(
                {
                    "asr": float(s.get("asr", 1.0)),
                    "family": s.get("family", "?"),
                    "severity": s.get("severity", "?"),
                }
            )
        groups.append(rows)
    return labels, groups


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    runs_dir = Path(args.runs_dir)
    run_ids = list(args.run) or sorted(
        p.name
        for p in runs_dir.iterdir()
        if (p / "config.json").exists()
        and json.loads((p / "config.json").read_text()).get("task") == "redteam"
    )
    if not run_ids:
        print("no redteam runs found", file=sys.stderr)
        return 1
    labels, groups = _rows_from_runs(runs_dir, run_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, order, fname, title in (
        ("family", FAMILIES, f"{args.prefix}-asr-family.png", "ASR per attack family"),
        ("severity", SEVERITIES, f"{args.prefix}-asr-severity.png", "ASR by severity"),
    ):
        xticks = [SHORT_FAMILY.get(g, g) if key == "family" else g for g in order]
        x = range(len(order))
        width = 0.8 / max(1, len(groups))
        _, ax = plt.subplots(figsize=(7, 4))
        for i, (label, rows) in enumerate(zip(labels, groups)):
            means = [
                statistics.mean([r["asr"] for r in rows if r[key] == g])
                if [r for r in rows if r[key] == g]
                else 0.0
                for g in order
            ]
            off = (i - (len(groups) - 1) / 2) * width
            bars = ax.bar([p + off for p in x], means, width, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(list(x), xticks)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("attack-success rate (higher = weaker)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = out_dir / fname
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"{sum(len(g) for g in groups)} samples -> {out}")
        for label, rows in zip(labels, groups):
            print(f"\n### {label}\n")
            print(_asr_table(rows, key, order))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Red-team runner + reporter (#22).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Eval one checkpoint; record; print ASR tables.")
    run.add_argument("--model", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--log-dir", default="logs")
    run.add_argument("--runs-dir", default="runs")
    plot = sub.add_parser("plot", help="ASR plots + tables from recorded runs.")
    plot.add_argument("--run", action="append", default=[])
    plot.add_argument("--runs-dir", default="runs")
    plot.add_argument("--out-dir", default="docs/reports")
    plot.add_argument("--prefix", default="issue-22-redteam")
    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(cmd_run(args))
    raise SystemExit(cmd_plot(args))


if __name__ == "__main__":
    main()
