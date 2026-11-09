# Tracking — Reproducibility Contract

## Sources of truth

1. **inspect-ai eval logs** (`.eval`): full transcripts, tool calls, scores.
   View with `inspect view`. Local dirs are redirected into the repo during
   sandboxed runs (`INSPECT_LOG_DIR`, `INSPECT_TRACE_FILE`, `XDG_DATA_HOME`;
   see [runbooks](runbooks.md)).
2. **`RunRegistry`** (`dmb/registry.py`): `runs/<run_id>/` holds
   `config.json` (task, model, scenario, seeds, limits, git sha),
   `scores.json`, `events.jsonl`, and eval-log pointers.
3. **Git**: code + scenario + asset versions. The registry records the HEAD
   sha of every run.

## Rules

- No result is real until a registry entry + eval log exist for it.
- Seeds are explicit and checked in with the task; `world_hash` proves the
  world a score came from.
- `runs/` and `logs/` are git-ignored working outputs; promote only
  curated summaries (tables in issues/PRs, golden scores in tests).
- Provider runs additionally record model slug + endpoint + price tier in
  the run notes so cost is auditable.

## Reading results

- Single run: `inspect view` on the `.eval` log.
- Across runs: `dmb/aggregate.py` (#8). Record each run once, then render:

```bash
python -m dmb.aggregate record --run-id <id> --task <task> --model <model-id> \
  --scenario <name-or-pack> --log <run.eval> [--note ...] \
  [--l2-dir runs/<id>/trajectories --l3-dir runs/<id>/sim]
python -m dmb.aggregate table --runs runs --out table.md [--csv out.csv]
python -m dmb.aggregate uplift --base <run> --sam <run>  # paired deltas
```

`record` writes `runs/<id>/{config.json,scores.json}` from the .eval
(per-sample vectors + eval-log pointer); `table` renders a
model×scenario markdown table (mean±std composite + full vector means,
pasteable into issues) and a per-sample CSV. Backfilled: `baseten-pack`,
`flytest-base`, `flytest-ft` (reproduces the posted #7 + flywheel numbers).
