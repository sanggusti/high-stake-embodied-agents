# Training runbook: curate → tune → eval → publish

Blind-followable loop for HSEA post-training (issue #17). Every command
below is quoted exactly as run from the repo root. Provenance:
[flywheel](flywheel.md) (v1 loop), [runbooks](runbooks.md) (envs),
[alignment-research](alignment-research.md) §5 (Lexsi-lib mapping),
`docs/reports/issue-7-flywheel-v2-curation.md` (v2 curation evidence).

Companion artifacts: card templates + the flywheel-v2 dataset card
draft live in [docs/cards/](cards/). The first-finetune **model**
card is out of scope (no finetune exists yet) — see the TODO in
[step 4](#4-publish) and
[cards/dmb-flywheel-v2-model-TODO.md](cards/dmb-flywheel-v2-model-TODO.md).

## 0. Environments (do first)

Two venvs. The default `.venv` is Python 3.10; the training env is
Python 3.12 (the only version satisfying every Lexsi lib). Never merge
them. If `uv` cannot reach its cache, prefix with
`UV_CACHE_DIR=/tmp/uv-cache`.

Main env (bench + evals + curation):

```bash
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
cp .env.example .env   # then fill in provider keys
```

Training env (pinned Lexsi libs, issue #26):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12 .venv-train
VIRTUAL_ENV=$PWD/.venv-train UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements-train.lock
.venv-train/bin/python scripts/smoke_train_env.py
```

Expected: `SMOKE PASSED (6/6 libs, pins exact)` with `aligntune
0.1.32, curatorkit 0.1.23, safetune 0.1.13, agenttune 1.1.3, auditkit
1.1.6, circuitkit 0.2.1` (verified on PyPI 2026-11-09). Transitive
set in the lock: `transformers 5.22.2`, `trl 1.7.1`,
`transformer-lens 3.8.0`, `torch 2.11.0`. GPU training itself runs on
Modal (CUDA images already `add_python("3.12")`); this venv proves
import/API compatibility only.

Keys (`.env`, never committed; see `.env.example`):

| Var | Use |
|---|---|
| `MODAL_TOKEN_ID` / `MODAL_SECRET_KEY` | Modal CLI auth (no `modal setup` needed) |
| `MODAL_BASE_URL` / `MODAL_API_KEY` | Serving route; URL must end in `/v1`; key any value |
| `HF_API_KEY` / `HF_BASE_URL` | HF Inference router (`https://router.huggingface.co/v1`) |
| `HUGGINGFACE_TOKEN` | Gated datasets/models pulls + HF publishes |
| `HSEA_VISION=text` | Text-only tool results for text teachers/navigators |

AGPL guardrails (enforced by `scripts/smoke_train_env.py`):

- AlignTune: TRL backend only. Never add the `aligntune[unsloth]`
  extra (AGPL); never `import unsloth` (the bare name resolves to a
  vendor shim that hard-crashes on macOS ARM).
- CuratorKIT: non-PDF ingest only (`JSONLReader` etc.). Never add
  `curatorkit[pdf]` (MinerU is AGPL-3.0).

## 1. Curate

### Path A — pinned corpora (flywheel-v2 method, `hsea/data/curate_v2.py`)

```bash
.venv/bin/python -m hsea.data.curate_v2 --out data/flywheel-v2 --split train
```

Repro gate (blind, no network, no writes — hashes + schemas + counts
+ clearance):

```bash
.venv/bin/python -m hsea.data.curate_v2 --out data/flywheel-v2 \
  --split train --verify-only
```

Defaults (override with `--max-*` flags; see `--help`): FloodNet-VQA
1000, LADI images 200, DisasterM3 3000, CrisiText situations 300,
HumAID 4000, CrisisMMD 3000, `--min-composite 0.5`,
`--l2-glob "logs/*.eval"`, `--v1-dir data/flywheel-v1`.
Clearance is asserted in-code before any streaming
(`hsea/data/clearance.py`, #24 terms); a non-cleared source aborts
the export. Output: `sft.jsonl` + `dpo.jsonl` + `rlvr.jsonl` +
`manifest.json` (code commit, revs, per-source counts/splits/filters,
SHA-256) under the `--out` dir. v2 totals: **8,673 SFT / 7,000 DPO /
250 RLVR** (see `data/flywheel-v2/manifest.json`).

### Path B — bench teacher episodes (flywheel-v1 method, `hsea/data/export.py`)

Teacher run on train seeds (forest_fire + tsunami × seeds 1..20 =
40 episodes). The sandbox blocks Modal's connection (DNS), so Modal
commands need an unsandboxed shell:

```bash
HSEA_VISION=text .venv/bin/inspect eval evals/flywheel_train.py \
  --model openai-api/modal/<teacher-name>
```

Then export (every episode records L2 `hsea/watch/` trajectories +
the L1 `.eval` log):

```bash
.venv/bin/python -m hsea.data.export --log <run.eval> \
  --l2 runs/<id>/trajectories --out data/<name> \
  --split train --min-composite 0.5
```

Split rule (`hsea/data/splits.py`): train seeds 1..20, held-out test
seeds 101..110, disjoint by construction. Split mixing is refused —
never train on test seeds.

### CuratorKIT gated-ingest touch (forward path)

The v2 exporter above is bespoke; CuratorKIT `JSONLReader` is the
verified gated-ingest API for its outputs (accepted, rejected):

```python
from curatorkit import JSONLReader

reader = JSONLReader("data/flywheel-v2/sft.jsonl")
samples, rejected = reader.read()  # gated curation: (accepted, rejected)
```

Non-PDF readers only (AGPL guardrail above).

## 2. Tune

### AlignTune (SFT + preference, TRL backend)

Verified API touch (from `scripts/smoke_train_env.py`):

```python
from aligntune import BackendFactory

avail = BackendFactory.list_available_backends()
assert "trl" in avail["SFT"]       # SFT via TRL
assert "trl" in avail["RL"]["dpo"]  # DPO via TRL
```

Recipe (per [alignment-research](alignment-research.md) §4): SFT,
then length-normalized DPO (KTO where only binary labels exist);
RLAIF self-critique vs the constitution to scale up. TRL backend
only — never the Unsloth extra.

### Concrete Modal LoRA path (v1-proven, `infra/modal/finetune.py`)

Needs `modal` auth; GPU spend applies. Upload the dataset, then run:

```bash
modal volume put dmb-flywheel data/flywheel-v1/sft.jsonl /flywheel-v1/sft.jsonl
modal run infra/modal/finetune.py
```

Output: `/models/flywheel-v1-merged/` on volume
`dmb-flywheel-models` (merged bf16 + tokenizer). Config (pinned in
the file): base `Qwen/Qwen2.5-7B-Instruct`, LoRA r16 / alpha 32 /
dropout 0.05 on q/k/v/o/gate/up/down, 3 epochs, batch 2 ×
grad-accum 4, lr 2e-4 cosine, warmup 5, bf16, max_length 4096,
packing off, seed 0, A10G. For a v2 finetune, point `DATASET_PATH`
at the v2 `sft.jsonl` (and DPO/RLVR stages at `dpo.jsonl` /
`rlvr.jsonl`) and rename `OUTPUT_DIR` — the file's constants are the
contract.

Serving gotchas (all hit live, see [flywheel](flywheel.md)): pin the
merged dir's tokenizer files to the base model's (transformers 5.x
schema breaks vLLM 0.21's v4 parser); serve with `--enforce-eager` +
20-min startup timeout (14 GB streams slowly off the volume);
Modal factory-pattern apps don't deploy (serve must be module-level).

### SafeTune (audit + repair)

Verified API surface (from `scripts/smoke_train_env.py`):

```python
from safetune import evaluate, harden

assert callable(evaluate.evaluate)
assert hasattr(harden, "AsFTTrainer")
```

Flow: `evaluate` pre/post AlignTune for drift; repair via a
safety-mix re-tune or constrained fine-tune (`harden` trainers);
check each method's Faithful badge first. Gate: AgentHarm harm-score
down, benign pass rate flat.

### AgentTune (agentic RL on tool traces)

Verified API surface (from `scripts/smoke_train_env.py`):

```python
from agenttune.agentic.events import EventLog
from agenttune.decide.collect_runner import CollectRunner
from agenttune.decide.graph_runner import GraphRunner

log = EventLog()  # the EventLog spine for tool-trace records
```

RLVR/GRPO on tool traces with safety-procedure rewards, then
red-team. Known-issue workarounds (AgentTune Known Issues page, read
2026-11-09): use `GraphRunner` for branching templates
(`StageWiseRunner` ignores conditional routing); use
`CollectRunner` for DPO/BCO records
(`AuditReader.extract_dpo_pairs` returns empty on real logs); avoid
`ToolRegistry` (crashes without `langchain_community`); confirm
`transformers`/`rouge_score`/`nltk` before trusting
DistilledJudge/Rouge/BLEU scores (all present in the lock);
construct registry `SQLDatabaseTool` with an explicit `db_uri`;
agentic tool-calling training only via the Python API (the CLI has
no `--tools` flag).

## 3. Eval

### Serve the finetune

```bash
modal deploy infra/modal/vllm_flywheel.py
```

Then set `MODAL_BASE_URL=<serve-url>/v1` (+ `MODAL_API_KEY`, any
value). The server is the v1 twin (`MODEL_PATH =
/models/flywheel-v1-merged`, hermes parser,
`--enable-auto-tool-choice`, `--enforce-eager`); point it at the new
merged dir for later finetunes.

### Held-out re-eval (base vs finetuned)

`evals/flywheel_test.py` runs held-out seeds 101..110. Same bench
conditions on both arms (including `HSEA_VISION=text` for text
policies):

```bash
HSEA_VISION=text .venv/bin/inspect eval evals/flywheel_test.py \
  --model openai-api/modal/<base-name>
HSEA_VISION=text .venv/bin/inspect eval evals/flywheel_test.py \
  --model openai-api/modal/<finetune-name>
```

Record each run once, then render the comparison table
(see [tracking](tracking.md)):

```bash
.venv/bin/python -m hsea.aggregate record --run-id <id> \
  --task flywheel_test --model <model-id> --scenario <pack> \
  --log <run.eval>
.venv/bin/python -m hsea.aggregate table --runs runs --out table.md
```

No result is real until a registry entry + eval log exist for it.
v1 reference: base mean 0.580, finetuned 0.531, delta −0.049 ≈ −1 SE
(statistically null — the scale the v2 prescription answers).

### Safety gates (re-run after *every* stage)

| Gate | Evals | Bar |
|---|---|---|
| 1 — refusal | HarmBench + AIR-Bench (external) | Must not drop vs base |
| 2 — agentic misuse | AgentHarm (external) + `evals/redteam.py` + `evals/safety.py` | Harm down, benign flat |
| 3 — embodied + escalation | SafeAgentBench (external) + `evals/abstention.py` + escalation curves | Pass + calibrated abstention |

In-repo harnesses (same invocation shape as above):

```bash
.venv/bin/inspect eval evals/safety.py --model <model-id>
.venv/bin/inspect eval evals/abstention.py --model <model-id>
.venv/bin/inspect eval evals/redteam.py --model <model-id>
```

Footage probes: `evals/floodnet_vqa.py` (validation split) and
`evals/ladi_v2.py` (test split) — disjoint from the v2 train rows by
construction. Per-release mini safety case: hazards → mitigations →
eval evidence → residual risks.

## 4. Publish

Pre-publish checklist (all must pass):

1. `--verify-only` green on the published data dir (hashes + schemas
   + counts + clearance).
2. No-leak checks for the v2 policy (labels/refs only):
   `tweet_text` absent from `dpo.jsonl`, `data:image` absent from
   `sft.jsonl`.
3. Rev pins verified live (upstream revs in `manifest.json` match the
   constants in `hsea/data/`; HF head rev recorded).
4. Dataset + model cards accurate (revs, configs, deltas) — see
   [docs/cards/](cards/).
5. HF uploads done (gated where NC/SA or ToS-bound); git tags
   `data-vX.Y` / `model-vX.Y` carry the HF revision pin; manifests +
   cards updated in the same PR (per [WORKFLOW](../WORKFLOW.md) §5).

Live artifacts:

| Artifact | HF id | Rev |
|---|---|---|
| flywheel-v1 dataset | `sanggusti/dmb-flywheel-v1` | vendored `data/flywheel-v1/` (commit `393a158`) |
| flywheel-v1 weights | `sanggusti/dmb-qwen2.5-7b-flywheel-v1` | merged bf16, tokenizer pinned to base |
| flywheel-v2 dataset | `sanggusti/dmb-flywheel-v2` (gated) | `b44237e145af1f0d9cfa3f22ec36f77d22ca5393` |

> **TODO (out of scope, #17 partial): first-finetune model card.**
> No v2 finetune exists yet, so there is no model card to publish.
> When the finetune lands: fill
> [cards/dmb-flywheel-v2-model-TODO.md](cards/dmb-flywheel-v2-model-TODO.md)
> from [cards/model-template.md](cards/model-template.md) (base rev,
> LoRA/data config, held-out delta table, safety-gate table, serving
> notes), publish weights + card to HF, tag `model-vX.Y`.

## Quick-command index

| Step | Command |
|---|---|
| Main env | `VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"` |
| Train env | `VIRTUAL_ENV=$PWD/.venv-train UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements-train.lock` |
| Train smoke | `.venv-train/bin/python scripts/smoke_train_env.py` |
| Curate (corpora) | `.venv/bin/python -m hsea.data.curate_v2 --out data/flywheel-v2 --split train` |
| Curate gate | `... --verify-only` (same command + flag) |
| Teacher run | `HSEA_VISION=text .venv/bin/inspect eval evals/flywheel_train.py --model openai-api/modal/<name>` |
| Export (L2) | `.venv/bin/python -m hsea.data.export --log <run.eval> --l2 runs/<id>/trajectories --out data/<name> --split train --min-composite 0.5` |
| Tune upload | `modal volume put dmb-flywheel data/flywheel-v1/sft.jsonl /flywheel-v1/sft.jsonl` |
| Tune run | `modal run infra/modal/finetune.py` |
| Serve FT | `modal deploy infra/modal/vllm_flywheel.py` |
| Held-out eval | `HSEA_VISION=text .venv/bin/inspect eval evals/flywheel_test.py --model openai-api/modal/<name>` |
| Record | `.venv/bin/python -m hsea.aggregate record --run-id <id> --task <t> --model <m> --scenario <s> --log <run.eval>` |
| Table | `.venv/bin/python -m hsea.aggregate table --runs runs --out table.md` |
| Safety | `.venv/bin/inspect eval evals/{safety,abstention,redteam}.py --model <model-id>` |
| Tests | `.venv/bin/python -m pytest -q` |
