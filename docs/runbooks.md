# Runbooks

## Setup (first clone)

```bash
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
cp .env.example .env   # then fill in provider keys
```

Requires Python ≥ 3.10. If `uv` cannot reach its cache, prefix commands with
`UV_CACHE_DIR=/tmp/uv-cache`.

## Training env (Python 3.12, Lexsi libs)

One pinned env installing all 6 Lexsi libs together (issue #26). Python 3.12
is the only version satisfying every lib (`safetune`/`agenttune` need ≥3.12).
The default `.venv` stays on 3.10 — do not merge the two.

Blind repro (docs only, from repo root):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12 .venv-train
VIRTUAL_ENV=$PWD/.venv-train UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements-train.lock
.venv-train/bin/python scripts/smoke_train_env.py
```

Expected: `SMOKE PASSED (6/6 libs, pins exact)` with
`aligntune 0.1.32, curatorkit 0.1.23, safetune 0.1.13, agenttune 1.1.3,
auditkit 1.1.6, circuitkit 0.2.1` (verified on PyPI 2026-11-09).
Transitive set in the lock: `transformers 5.22.2`, `trl 1.7.1`,
`transformer-lens 3.8.0`, `torch 2.11.0`.

AGPL guardrails (enforced by the smoke script):

- AlignTune: TRL backend only (`BackendFactory` lists `trl` for SFT + all RL
  algos). Never add the `aligntune[unsloth]` extra — the AGPL `unsloth`
  distribution must stay absent (`importlib.metadata.version("unsloth")`
  raises `PackageNotFoundError`). Note: the bare name `unsloth` resolves to
  a vendor shim bundled with aligntune/agenttune; never `import unsloth`
  (it hard-crashes on macOS ARM) — and AlignTune itself reports it
  unavailable.
- CuratorKIT: non-PDF ingest only (`JSONLReader` etc.). Never add
  `curatorkit[pdf]` — MinerU is AGPL-3.0.

Known-issue workarounds (AgentTune Known Issues page, read 2026-11-09;
no new blockers filed — all worked around):

1. `ToolRegistry.get()` crashes without `langchain_community` → the smoke
   script avoids `ToolRegistry` entirely.
2. `DistilledJudge`/`RougeMetric`/`BleuMetric` silently return 0 without
   `transformers`/`rouge_score`/`nltk` → confirm those deps before trusting
   scores (all three are present in this lock).
3. `StageWiseRunner` ignores conditional routing → use `GraphRunner` for
   branching templates (import-checked in the smoke script).
4. `AuditReader.extract_dpo_pairs` returns empty on real logs → use
   `CollectRunner` for DPO/BCO records (import-checked).
5. The CLI has no `--tools` flag → agentic tool-calling training only via
   the Python API.
6. Registry `SQLDatabaseTool` ships unconfigured → construct with an
   explicit `db_uri`.

Notes: the `aligntune ... Missing credentials` warning during the smoke run
is benign (no `OPENAI_API_KEY` needed for the TRL smoke path). Import order
is unconstrained in this env (circuitkit-first and circuitkit-last both
verified). GPU training itself runs on Modal (CUDA images already
`add_python("3.12")`); this venv proves import/API compatibility only.

## Daily commands

```bash
.venv/bin/python -m pytest -q                          # unit tests
.venv/bin/inspect eval evals/smoke_inference.py --model mockllm/model
XDG_DATA_HOME=$PWD/.inspect-data INSPECT_LOG_DIR=./logs \
INSPECT_TRACE_FILE=./logs/trace.log \
  .venv/bin/inspect eval evals/forest_fire.py --model mockllm/model --limit 1
```

Provider runs (keys required):

```bash
.venv/bin/inspect eval evals/forest_fire.py \
  --model openai-api/baseten/<slug> --limit 1
.venv/bin/inspect eval evals/forest_fire.py \
  --model openai-api/modal/<name> --limit 1
```

Sim-backend runs are retired in HSEA: `HSEA_SIM_BACKEND` takes only `mock`
(the Gazebo/SITL + Docker workstream stayed in the bench repo).

Baselines (provider-free, deterministic):

```bash
.venv/bin/python -m pytest tests/test_baselines.py -q
.venv/bin/inspect eval evals/baselines.py --model mockllm/model
```

Human baseline mission (blind; scored + recorded like any run):

```bash
.venv/bin/python -m hsea.human_cli --scenario forest_fire --seed 1 --name <you>
```

## Replay

```bash
# T1 narrative + T2 mock render + T3 composite page for one episode:
.venv/bin/python -m hsea.watch.replay runs/<id>/trajectories/<sample>.jsonl
.venv/bin/python -m hsea.watch.render runs/<id>/trajectories/<sample>.jsonl --out runs/<id>/sim/<sample>
.venv/bin/python -m hsea.watch.compose --eval <run.eval> --sample <sample> \
  --l2 runs/<id>/trajectories --l3 runs/<id>/sim --out runs/<id>/sim/<sample>.html
# (T2 Gazebo state-log playback retired with the sim image.)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: hsea` | Use `.venv/bin/...` or `pip install -e .` |
| `PermissionError .../Application Support/inspect_ai` | Export the three log-dir vars above (sandboxed shells) |
| `Cannot import 'hatchling.build'` | Backend is setuptools; pull latest `pyproject.toml` |
| Provider auth errors | Check `<SERVICE>_API_KEY/_BASE_URL` for the service segment in the model id |
| Modal 404 on `/v1` | `MODAL_BASE_URL` must include the `/v1` suffix |

## Publishing flow (required)

Issues first, then PR, then merge, then close. No direct pushes to `main`:

1. File/extend a GitHub issue describing the work.
2. Branch `feat/<slug>`, commit named files only.
3. `gh pr create` referencing the issue; review; `gh pr merge`.
4. Verify linked issues auto-close; close leftovers by hand with a note.
