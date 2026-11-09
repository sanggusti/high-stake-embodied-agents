# Runbooks

## Setup (first clone)

```bash
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
cp .env.example .env   # then fill in provider keys
```

Requires Python ≥ 3.10. If `uv` cannot reach its cache, prefix commands with
`UV_CACHE_DIR=/tmp/uv-cache`.

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
