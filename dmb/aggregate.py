"""Results aggregation (#8): eval logs -> registry -> tables.

Two steps (registry entries were specified but never written, so this both
records and aggregates):

- record: ``python -m dmb.aggregate record --run-id <id> --task <task>
  --model <model-id> --scenario <name-or-pack> --log <run.eval>
  [--note ...]`` writes ``runs/<id>/{config.json,scores.json}`` from the
  .eval (per-sample metric vectors + eval-log pointer).
- aggregate: ``python -m dmb.aggregate table --runs runs --out table.md
  [--csv out.csv]`` renders a markdown table (model x scenario means)
  plus a per-sample CSV with full vectors.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path

from dmb.registry import RunRegistry

VECTOR_KEYS = (
    "rescue_recall",
    "containment",
    "efficiency",
    "safety",
    "composite",
)

ROUTE_KEYS = (
    "routes_requested",
    "routes_reachable_rate",
    "route_safety",
    "route_excess_length",
    "spl_like",
)


def record_run(
    run_id: str,
    task: str,
    model: str,
    scenario: str,
    log_path: str | Path,
    runs_dir: str | Path = "runs",
    notes: list[str] | None = None,
    l2_dir: str | Path | None = None,
    l3_dir: str | Path | None = None,
) -> dict:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    log = read_eval_log(str(log_path))
    registry = RunRegistry(runs_dir)
    samples = []
    seeds: list[int] = []
    for sample in log.samples or []:
        meta = dict(sample.metadata or {})
        seed = meta.get("seed")
        if isinstance(seed, int) and seed not in seeds:
            seeds.append(seed)
        score = (sample.scores or {}).get("mission_score")
        samples.append(
            {
                "sample_id": sample.id,
                "scenario": meta.get("scenario"),
                "seed": seed,
                "world_hash": meta.get("world_hash"),
                "scores": dict(getattr(score, "metadata", None) or {}),
                "error": sample.error.message if sample.error else None,
            }
        )
    record = registry.start_run(
        run_id, task=task, model=model, scenario=scenario, seeds=sorted(seeds)
    )
    # Registry links all three evidence layers (#16).
    links = {"l1_eval_log": str(log_path)}
    if l2_dir is not None:
        links["l2_trajectories"] = str(l2_dir)
    if l3_dir is not None:
        links["l3_sim"] = str(l3_dir)
    record.notes.extend(notes or [])
    config_path = Path(runs_dir) / run_id / "config.json"
    payload = json.loads(config_path.read_text())
    payload["notes"] = record.notes
    payload["evidence"] = links
    config_path.write_text(json.dumps(payload, indent=2, default=str))
    registry.write_scores(run_id, {"samples": samples})
    registry.finish_run(run_id, eval_logs=[str(log_path)])
    return {"run_id": run_id, "samples": len(samples)}


def scores_from_l2(l2_path: str | Path) -> dict:
    """Extract the scorer's metric vector from an L2 log (audit rule).

    The mission scorer appends its breakdown as the ``score`` event, so
    L2 alone reproduces the sample's scores without the .eval log.
    """
    for line in Path(l2_path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("kind") == "score":
            return {k: v for k, v in event.items() if k not in {"t", "step", "kind"}}
    raise ValueError(f"no score event in {l2_path}")


def _load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for run_path in sorted(runs_dir.iterdir()):
        config_path = run_path / "config.json"
        scores_path = run_path / "scores.json"
        if not (config_path.exists() and scores_path.exists()):
            continue
        config = json.loads(config_path.read_text())
        scores = json.loads(scores_path.read_text())
        for sample in scores.get("samples", []):
            rows.append(
                {
                    "run_id": config.get("run_id", run_path.name),
                    "task": config.get("task", ""),
                    "model": config.get("model", ""),
                    "scenario": sample.get("scenario") or config.get("scenario", ""),
                    "seed": sample.get("seed"),
                    "sample_id": sample.get("sample_id"),
                    **{
                        k: (sample.get("scores") or {}).get(k)
                        for k in (*VECTOR_KEYS, *ROUTE_KEYS)
                    },
                }
            )
    return rows


def _mean_std(values: list[float]) -> str:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:.3f}"
    return f"{statistics.mean(clean):.3f}±{statistics.stdev(clean):.3f}"


DISPLAY_KEYS = ("composite", "rescue_recall", "containment", "efficiency", "safety")
DISPLAY_HEADERS = ("composite", "recall", "contain", "effic", "safety")


def render_markdown(rows: list[dict]) -> str:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["scenario"]), []).append(row)
    lines = [
        "| model | scenario | n | " + " | ".join(DISPLAY_HEADERS) + " |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (model, scenario), group in sorted(groups.items()):
        short = model.split("/")[-1]
        cells = [short, scenario, str(len(group))]
        for key in DISPLAY_KEYS:
            cells.append(_mean_std([r[key] for r in group]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def uplift_markdown(base_id: str, sam_id: str, runs_dir: str | Path) -> str:
    """Paired per-sample deltas (sam - base) over matched samples."""
    rows = _load_rows(Path(runs_dir))

    def keyed(run_id: str) -> dict:
        return {
            (r["scenario"], r["seed"], r["sample_id"]): r
            for r in rows if r["run_id"] == run_id
        }

    base, sam = keyed(base_id), keyed(sam_id)
    paired = [k for k in base if k in sam]
    if not paired:
        raise ValueError(f"no paired samples between {base_id!r} and {sam_id!r}")
    lines = [
        f"| metric | base mean | sam mean | delta mean | n |",
        "|---|---|---|---|---|",
    ]
    for metric in (*VECTOR_KEYS, *ROUTE_KEYS):
        deltas, bases, sams = [], [], []
        for k in paired:
            b, s = base[k].get(metric), sam[k].get(metric)
            if isinstance(b, (int, float)) and isinstance(s, (int, float)):
                bases.append(b)
                sams.append(s)
                deltas.append(s - b)
        if not deltas:
            cells = ["n/a", "n/a", "n/a", "0"]
        else:
            cells = [
                f"{statistics.mean(bases):.3f}",
                f"{statistics.mean(sams):.3f}",
                f"{statistics.mean(deltas):+.3f}",
                str(len(deltas)),
            ]
        lines.append("| " + " | ".join([metric, *cells]) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["run_id", "task", "model", "scenario", "seed", "sample_id", *VECTOR_KEYS],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record runs; aggregate tables.")
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="Write a registry entry from an .eval log.")
    record.add_argument("--run-id", required=True)
    record.add_argument("--task", required=True)
    record.add_argument("--model", required=True)
    record.add_argument("--scenario", required=True)
    record.add_argument("--log", required=True)
    record.add_argument("--runs", default="runs")
    record.add_argument("--note", action="append", default=[])
    record.add_argument("--l2-dir", default=None)
    record.add_argument("--l3-dir", default=None)
    table = sub.add_parser("table", help="Render markdown + CSV from the registry.")
    table.add_argument("--runs", default="runs")
    table.add_argument("--out", required=True)
    table.add_argument("--csv", default=None)
    table.add_argument("--run", action="append", default=[], help="Only include these run ids (repeatable).")
    uplift = sub.add_parser("uplift", help="Paired sam-minus-base delta table.")
    uplift.add_argument("--base", required=True)
    uplift.add_argument("--sam", required=True)
    uplift.add_argument("--runs", default="runs")
    args = parser.parse_args()
    if args.command == "record":
        print(
            json.dumps(
                record_run(
                    args.run_id, args.task, args.model, args.scenario,
                    args.log, args.runs, args.note, args.l2_dir, args.l3_dir,
                ),
                indent=2,
            )
        )
    elif args.command == "uplift":
        print(uplift_markdown(args.base, args.sam, args.runs), end="")
    else:
        rows = _load_rows(Path(args.runs))
        if args.run:
            rows = [r for r in rows if r["run_id"] in set(args.run)]
        Path(args.out).write_text(render_markdown(rows))
        print(f"{len(rows)} samples -> {args.out}")
        if args.csv:
            Path(args.csv).write_text(render_csv(rows))
            print(f"{len(rows)} samples -> {args.csv}")


if __name__ == "__main__":
    main()
