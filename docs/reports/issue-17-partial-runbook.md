# Issue #17 (partial) — training runbook + cards, no finetune yet

Branch: `issue-17` (pushed, no PR — partial by design). Date: 2026-11-10.

## What

Blind-followable curate → tune → eval → publish runbook
(`docs/training-runbook.md`) quoting exact commands/env/paths for the
pinned train env (`requirements-train.lock` +
`scripts/smoke_train_env.py`); reusable card templates
(`docs/cards/`); flywheel-v2 dataset card content (rev `b44237e`,
manifest counts). The first-finetune model card is out of scope — no
finetune exists yet (TODO stub committed). HF publishes remain.

## Commands quoted (each verified against the repo)

| # | Command / snippet | Verified against |
|---|---|---|
| 1 | `uv venv .venv` + `uv pip install -e ".[dev]"` | `docs/runbooks.md` Setup |
| 2 | `uv venv --python 3.12 .venv-train` + lock install | `docs/runbooks.md` Training env |
| 3 | `.venv-train/bin/python scripts/smoke_train_env.py` → `SMOKE PASSED (6/6 libs, pins exact)` | script + live run (this issue) |
| 4 | 6 lib pins + 4 transitive pins | `requirements-train.txt` + `.lock` + live smoke |
| 5 | `hsea.data.curate_v2 --out … --split train [--verify-only]` + `--max-*` defaults | `hsea/data/curate_v2.py` argparse + `--help` + live `--verify-only` |
| 6 | 7 upstream revs (floodnet/ladi/disasterm3/crisitext/humaid/crisismmd/v2) | `hsea/data/footage.py` + `curate_v2.py` constants vs `manifest.json` |
| 7 | `HSEA_VISION=text … inspect eval evals/flywheel_train.py --model openai-api/modal/<name>` | `evals/flywheel_train.py` + `docs/inference.md` routing + `hsea/tools/drone.py` |
| 8 | `hsea.data.export --log --l2 --out --split --min-composite` | `hsea/data/export.py` argparse + docstring |
| 9 | Train seeds 1..20 / test 101..110, disjoint | `hsea/data/splits.py` |
| 10 | `BackendFactory.list_available_backends()` TRL SFT+DPO | `scripts/smoke_train_env.py` |
| 11 | `modal volume put dmb-flywheel …` + `modal run infra/modal/finetune.py` + LoRA config | `infra/modal/finetune.py` docstring + constants |
| 12 | `safetune.evaluate.evaluate` + `harden.AsFTTrainer` | `scripts/smoke_train_env.py` |
| 13 | `EventLog` + `GraphRunner` + `CollectRunner` + 6 known-issue avoids | `scripts/smoke_train_env.py` + `docs/runbooks.md` |
| 14 | `modal deploy infra/modal/vllm_flywheel.py` + `MODAL_BASE_URL=<url>/v1` | `infra/modal/vllm_flywheel.py` + `docs/inference.md` |
| 15 | `inspect eval evals/flywheel_test.py` (base + FT) + `aggregate record/table` | `evals/flywheel_test.py` + `hsea/aggregate.py` CLI + `docs/tracking.md` |
| 16 | `inspect eval evals/{safety,abstention,redteam}.py` + gates table | eval `@task` names + `docs/alignment-research.md` §4 |
| 17 | v1 delta 0.580 → 0.531 (−0.049 ≈ −1 SE) | `docs/flywheel.md` |
| 18 | v2 totals 8,673 / 7,000 / 250 (133/117) + SHA-256 | `manifest.json` + live recount of `rlvr.jsonl` |
| 19 | HF rev `b44237e145af1f0d9cfa3f22ec36f77d22ca5393` | `data/flywheel-v2/README.md` + issue-7 report |
| 20 | `.venv/bin/python -m pytest -q` green | live full-suite run (this issue) |

## Cards status

| Card | File | Status |
|---|---|---|
| Dataset template | `docs/cards/dataset-template.md` | Committed (all `{{PLACEHOLDER}}`s) |
| Model template | `docs/cards/model-template.md` | Committed (all `{{PLACEHOLDER}}`s) |
| Flywheel-v2 dataset content | `docs/cards/dmb-flywheel-v2-dataset.md` | Drafted, counts verified — Hub upload REMAINS |
| First-finetune model | `docs/cards/dmb-flywheel-v2-model-TODO.md` | TODO stub — finetune + card + upload REMAIN |

## Why no plots

Docs-only change with no new numeric results: the only plottable
counts in scope are flywheel-v2's, already plotted by
`scripts/plot_flywheel_v2.py` in the issue-7 report
(`docs/reports/issue-7-flywheel-v2-curation.png`). Re-rendering the
identical figure would duplicate evidence, not add it.

## Verification runs

- `curate_v2 --verify-only` on `data/flywheel-v2`: passes (hashes +
  schemas + counts + clearance).
- `scripts/smoke_train_env.py` in `.venv-train`: `SMOKE PASSED
  (6/6 libs, pins exact)` — every lib pin and API touch quoted in
  the runbook re-verified live.
- Link check: every relative link in the new docs resolves
  (`training-runbook.md` ↔ `cards/`, README ↔ `docs/`).
- Full suite `.venv/bin/python -m pytest -q`: green (docs-only
  change; suite must stay green).

## Cost

- Estimate: $0.00 — docs-only, no Modal, no provider calls.
- Actual: $0.00. Ledger row `17 runbook-docs cpu 0.1 $0.00`
  appended to `/tmp/modal-spend.log` and mirrored in
  `docs/reports/modal-spend.md`.
- Running TOTAL after this issue: **$0.50** (unchanged).

## Artifacts

- `docs/training-runbook.md` (+ links from `README.md`, `docs/README.md`)
- `docs/cards/` (index, 2 templates, v2 dataset card, model TODO stub)
- This report.

## Shutdown

No Modal apps started — nothing to stop. No GPU spend.

## Remains (follow-ups, not this branch)

1. Run the first v2 finetune (SFT → DPO/KTO → SafeTune → AgentTune).
2. Fill + publish the first-finetune model card; tag `model-vX.Y`.
3. Upload the drafted v2 dataset card to
   `sanggusti/dmb-flywheel-v2` (data is live at `b44237e`; the Hub
   card copy lags).
