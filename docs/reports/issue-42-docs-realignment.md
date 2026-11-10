# Issue #42: Docs realignment, HSEA research track plus Lexsi process

Branch: `issue-42`. Date: 2026-11-10.

## What

Deleted five sim-era doc stubs, repaired every inbound link, and added
two HSEA docs (research track plus operating procedure) with updated
indexes. Docs and links only; code, data, configs, and env files are
untouched.

Deletion table (`git rm` from `origin/main`):

| Deleted | Lines | Why |
|---|---|---|
| `docs/scenarios.md` | 59 | Sim scenario catalog; evals compose from datasets now |
| `docs/roadmap.md` | 59 | Sim-era workstream list; planning lives in issues |
| `docs/peatforest-evals.md` | 64 | Peat sim eval page; FIRMS peat scope retired |
| `docs/flywheel.md` | 80 | Sim-data loop doc; Lexsi loop replaces it |
| `docs/action-surface.md` | 64 | ArduPilot/GUIDED surface; sim flight stacks stayed behind |

Line counts from `git show origin/main:<path> | wc -l`.

## Link-repair log

Grep pattern (tracked `.md` and `.py` files):

```text
scenarios\.md|roadmap\.md|peatforest-evals\.md|flywheel\.md|action-surface\.md
```

Before (5 hits): 4 index rows in `docs/README.md`
(action-surface, scenarios, flywheel, roadmap) plus 1 live pointer in
`docs/reports/issue-7-flywheel-v2-curation.md`
("per the flywheel.md v2 prescription"). `docs/peatforest-evals.md`
had no inbound links outside its own file.

After repair (0 hits outside this report): `docs/README.md` drops the
4 rows and indexes the new docs; the issue-7 report now reads as
history ("per the v1 loop prescription from the retired flywheel doc"
with a pointer to `docs/research-track.md`). `SEEDING.md` keeps its
seeding-manifest mention of the retired names without the `.md`
suffix; that file is a historical record of what was seeded.

This report names the deleted files in its tables; those mentions are
the historical record allowed for `docs/reports/`.

Flywheel model unlink: `sanggusti/dmb-qwen2.5-7b-flywheel-v1` appeared
once (weights link in the deleted `docs/flywheel.md`). After deletion
the id appears nowhere except this log row (historical record, no
URL). The v1 dataset example at `data/flywheel-v1/` stays referenced
as a format example.

## New docs

`docs/research-track.md`: HSEA big picture. Mission (safety-graded
finetuning loop, DORA-inspired, simulator-free); dataset-composed
footage, action, and safety tracks drawn only from `docs/datasets.md`
(smoke checks routing only and never counts as a result); model lanes
drawn only from `docs/models.md` (Cohere, Qwen, Molmo2, Gemma, HF
Inference frontier, edge; Qwen2.5 and Mistral-7B-v0.3 cited as legacy
only); curate to eval loop diagram; retired table with reasons; one
honest-null note on the v1 loop with no sim specifics.

`docs/process.md`: Lexsi-loop operating procedure. CuratorKIT export
(command, clearance gate, manifest), AlignTune SFT (3.12 train env
from `requirements-train.lock`, `scripts/smoke_train_env.py`, Modal
LoRA path), SafeTune drift gates (thresholds from
`docs/safety-gates.md`, PR wiring), AgentTune loop (L2 inputs,
before/after metrics), inspect-ai eval, record, table, and plot
conventions (aggregate CLI plus footage, safety, abstention, and
redteam scorers), branch to PR to merge pointers to `WORKFLOW.md`.
Every command was verified against the repo (pins, paths, flags).

Index updates: `README.md` gains research-track plus process entry
points; `docs/README.md` drops the 4 deleted rows, adds the 2 new
docs, and indexes 6 previously unindexed surviving docs
(footage-evals, abstention-evals, redteam, safety-gates,
license-clearance-24), with the stale sim-era header reframed on
dataset-composed tracks.

Style compliance for all new prose: HSEA title throughout, no
legacy repo naming in prose, no em-dashes, no contrastive AI-tell
phrasing, zero eval scores cited (nothing to invent; scores live in
their own eval docs and reports).

## Verification

* Filename grep: 5 hits before, 0 after (outside this report).
* Model id grep: 1 hit before (deleted file), 0 after (outside this
  log row).
* Full suite via the main-repo venv python:
  `.venv/bin/python -m pytest -q` gives 244 passed, 7 skipped.
* New-doc link check: every relative link in `research-track.md`,
  `process.md`, `README.md`, and `docs/README.md` resolves to a file
  present on this branch (checked by listing `docs/` plus root).
* Command check: each command in `process.md` was read back against
  its source file or `--help` surface (aggregate subcommands and
  flags, curate flags and manifest keys, smoke pins, finetune volume
  paths and LoRA config, safety and redteam and abstention script
  flags, watcher replay commands).

## Plots

None. This issue runs no evals and produces no scores, so there is
nothing to distribute or compare. A chart over deletion counts would
be chartjunk; the tables above carry the full record.

## Cost

* Estimate (pre-run): $0.00. Docs-only issue: file edits, grep, and
  a local pytest run. No model calls, no deploys.
* Actual: $0.00. Ledger row `42 docs-realign cpu 0.2 $0.00` appended
  to `/tmp/modal-spend.log`; `docs/reports/modal-spend.md` updated by
  union (all log rows mirrored).
* Running TOTAL after this issue: **$2.78** (unchanged; $0 added).

## Artifacts

* Deleted: the 5 files in the table above.
* Added: `docs/research-track.md`, `docs/process.md`, this report.
* Edited: `docs/README.md`, `README.md`,
  `docs/reports/issue-7-flywheel-v2-curation.md`,
  `docs/reports/modal-spend.md`.
* Verification logs: `/tmp/issue42-grep-before.txt`,
  `/tmp/issue42-grep-after.txt` (session-local, not committed).

## Shutdown

No Modal apps started. No GPU spend. No PR opened, no merge, issue #42
left open per the task.
