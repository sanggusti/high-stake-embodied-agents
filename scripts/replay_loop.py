"""Fixed-seed replay loop: bundles + reproduced comparison (issue #8).

Replay deliverable (user order): per-stage before/after example pairs
in the report, saved replay bundles, and a fixed-seed script
reproducing the comparison from raw logs.

- ``build``: assemble ``runs/issue-8/replays/<track>.jsonl`` bundles
  (inputs, per-stage generations, judge reasoning+verdicts, seeds)
  from eval logs + judgments. Seed 8 everywhere.
- ``verify``: reproduce the comparison deterministically — recompute
  summary.json, drift tables, and the report's headline numbers from
  raw .eval logs + judgments, and diff against the committed files.
  Non-zero exit on any mismatch.
- ``upload``: push bundles to HF (``sanggusti/dmb-issue-8-replays``).

Usage:
  .venv/bin/python scripts/replay_loop.py --issue 8 build
  .venv/bin/python scripts/replay_loop.py --issue 8 verify
  .venv/bin/python scripts/replay_loop.py --issue 8 upload [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 8
REPLAY_REPO = "sanggusti/dmb-issue-8-replays"
STAGES = ("base", "sft", "dpo", "safetune", "grpo", "rft")
TRACKS = ("flywheel_test", "floodnet_vqa", "ladi_v2", "safety", "redteam")


def _run_dir(issue: int) -> Path:
    return ROOT / "runs" / f"issue-{issue}"


def _flatten(log_path: str) -> list[dict]:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    log = read_eval_log(log_path)
    rows = []
    for s in log.samples or []:
        score = list((s.scores or {}).values())[0] if s.scores else None
        rows.append({
            "id": str(s.id),
            "input": s.input if isinstance(s.input, str) else str(s.input),
            "completion": (s.output.completion or "")
            if s.output is not None else "",
            "target": str(getattr(s.target, "text", s.target) or ""),
            "metadata": dict(s.metadata or {}),
            "value": float(score.value) if score is not None else 0.0,
            "score_metadata": dict(getattr(score, "metadata", None) or {})
            if score else {},
        })
    return rows


def cmd_build(issue: int) -> None:
    """Assemble per-track replay bundles (seed 8, deterministic)."""
    run_dir = _run_dir(issue)
    manifest = json.loads((run_dir / "eval_manifest.json").read_text())
    judgments: dict[str, dict] = {}
    judgments_path = run_dir / "judgments.jsonl"
    if judgments_path.exists():
        for line in open(judgments_path):
            row = json.loads(line)
            judgments.setdefault(row["key"], {})[row["judge"]] = {
                "text": row["text"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
            }
    replay_dir = run_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for track in TRACKS:
        bundle: list[dict] = []
        stages_present = [st for st in STAGES
                          if f"{st}/{track}" in manifest["runs"]]
        if not stages_present:
            continue
        by_stage = {st: _flatten(manifest["runs"][f"{st}/{track}"]["log"])
                    for st in stages_present}
        ids = [r["id"] for r in by_stage[stages_present[0]]]
        for rid in ids:
            rec: dict = {"seed": SEED, "track": track, "id": rid,
                         "stages": {}}
            for st in stages_present:
                match = next((r for r in by_stage[st] if r["id"] == rid),
                             None)
                if match is None:
                    continue
                cell = {"input": match["input"],
                        "completion": match["completion"],
                        "target": match["target"],
                        "value": match["value"],
                        "metadata": match["metadata"]}
                key = None
                for tmpl in ("T1", "T2", "T4"):
                    cand = f"{tmpl}:{st}:{track}:{rid}"
                    if cand in judgments:
                        key = cand
                        break
                if key:
                    cell["judge"] = judgments[key]
                rec["stages"][st] = cell
            bundle.append(rec)
        path = replay_dir / f"{track}.jsonl"
        with open(path, "w") as fh:
            for rec in bundle:
                fh.write(json.dumps(rec) + "\n")
        counts[track] = len(bundle)
        print(f"bundle {track}: {len(bundle)} records -> {path}")
    (replay_dir / "manifest.json").write_text(json.dumps(
        {"issue": issue, "seed": SEED, "tracks": counts}, indent=2))


def _recompute_summary(run_dir: Path) -> dict:
    """Recompute summary.json from raw logs (independent path)."""
    import issue8_evals as ev  # noqa: PLC0415

    manifest = json.loads((run_dir / "eval_manifest.json").read_text())
    summary: dict = {"runs": {}}
    for key, run in manifest["runs"].items():
        alias = key.split("/", 1)[1]
        rows = ev._flatten_samples(run["log"], run["scorer"])
        summary["runs"][key] = {
            "log": run["log"],
            "metrics": ev._SUMMARIZERS[alias](rows),
        }
    return summary


def cmd_verify(issue: int) -> None:
    """Reproduce the comparison from raw logs; diff vs committed."""
    run_dir = _run_dir(issue)
    failures: list[str] = []
    committed = json.loads((run_dir / "summary.json").read_text())
    fresh = _recompute_summary(run_dir)
    if committed.keys() != fresh.keys():
        failures.append(f"summary top-level keys differ: "
                        f"{sorted(committed)} vs {sorted(fresh)}")
    for key in committed.get("runs", {}):
        if key not in fresh["runs"]:
            failures.append(f"missing recomputed run {key}")
            continue
        want = json.dumps(committed["runs"][key]["metrics"], sort_keys=True)
        got = json.dumps(fresh["runs"][key]["metrics"], sort_keys=True)
        if want != got:
            failures.append(f"metrics differ for {key}")
    # Judge roll-up reproducibility (pure function over committed I/O).
    judge_path = run_dir / "judge_summary.json"
    if judge_path.exists():
        import issue8_judge as judge_mod  # noqa: PLC0415

        inputs = [json.loads(line)
                  for line in open(run_dir / "judge_inputs.jsonl")]
        judgments: dict[str, dict[str, str]] = {}
        for line in open(run_dir / "judgments.jsonl"):
            row = json.loads(line)
            judgments.setdefault(row["key"], {})[row["judge"]] = row["text"]
        fresh_j = judge_mod.analyze_inputs(inputs, judgments)
        committed_j = json.loads(judge_path.read_text())
        if (json.dumps(fresh_j, sort_keys=True)
                != json.dumps(committed_j, sort_keys=True)):
            failures.append("judge_summary.json not reproducible")
    if failures:
        for fail in failures:
            print(f"MISMATCH: {fail}")
        raise SystemExit(f"replay verify FAILED ({len(failures)} mismatches)")
    print(f"replay verify OK: {len(committed.get('runs', {}))} runs + "
          f"judge roll-up reproduce exactly (seed {SEED})")


def _client():
    from huggingface_hub import HfApi

    token = (os.environ.get("HF_API_KEY")
             or os.environ.get("HUGGINGFACE_TOKEN")
             or os.environ.get("HF_TOKEN"))
    if not token:
        raise SystemExit("missing HF_API_KEY (or HUGGINGFACE_TOKEN)")
    return HfApi(token=token)


def cmd_upload(issue: int, dry_run: bool = False) -> None:
    """Upload replay bundles to HF (never prints the token)."""
    run_dir = _run_dir(issue)
    replay_dir = run_dir / "replays"
    files = sorted(replay_dir.glob("*.jsonl")) + [replay_dir / "manifest.json"]
    card = f"""---
license: other
tags:
- replay
- disaster-response
---

# Issue #{issue} replay bundles (seed {SEED})

Per-track before/after records: inputs, per-stage generations, judge
reasoning + verdicts. Reproduced by
``scripts/replay_loop.py --issue {issue} verify``.
"""
    if dry_run:
        print(f"dry-run: would upload {len(files)} files to {REPLAY_REPO}")
        return
    api = _client()
    api.create_repo(REPLAY_REPO, exist_ok=True, private=True,
                    repo_type="dataset")
    for path in files:
        api.upload_file(repo_id=REPLAY_REPO, repo_type="dataset",
                        path_in_repo=f"issue-{issue}/{path.name}",
                        path_or_fileobj=str(path),
                        commit_message=f"issue-{issue} replay {path.name}")
    api.upload_file(repo_id=REPLAY_REPO, repo_type="dataset",
                    path_in_repo="README.md", path_or_fileobj=card.encode(),
                    commit_message=f"issue-{issue} replay card")
    print(f"uploaded {len(files)} files to {REPLAY_REPO}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("command", choices=["build", "verify", "upload"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        cmd_build(args.issue)
    elif args.command == "verify":
        cmd_verify(args.issue)
    elif args.command == "upload":
        cmd_upload(args.issue, args.dry_run)


if __name__ == "__main__":
    main()
