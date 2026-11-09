# Footage-grounded VLM eval results (#19)

Tasks: `evals/floodnet_vqa.py` + `evals/ladi_v2.py` — real footage corpora,
no sim. Scorer: `footage_score` (deterministic; value = routed metric,
full vector in metadata — see [metrics](metrics.md)).

## Pinned sources (verified 2026-11-09)

| Corpus | HF id | Revision | Split | License |
|---|---|---|---|---|
| FloodNet-VQA | `ljx620/FloodNet-VQA` | `ec9277af` | validation | untagged on card (annotations derive from FloodNet CDLA-Permissive); training clarified by #7 (QA text + refs only, gated research-only) |
| LADI v2 | `MITLL/LADI-v2-dataset` | `5f2dbfe8` | test | CC-BY-4.0 (official MITLL upload) |

Notes: `torchgeo/floodnet` carries images/seg only (no VQA pairs);
`FrancescoCiccone/FloodNet_VQA` is gated (auto-approval) with unknown
license, so the ungated `ljx620` webdataset mirror is pinned instead.
LADI labels render as 12 yes/no questions per image; the 6 damage-grade
columns route to `severity_f1`. Grounding (`grounding_acc`, grid-cell
match) is scored and unit-tested but has no corpus task yet — the
DisasterM3 referring subset is the planned follow-up.

## Harness smoke (mockllm, this PR)

| model | scenario | n | composite | vqa | count | severity | ground |
|---|---|---|---|---|---|---|---|
| mockllm | floodnet-vqa | 2 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| mockllm | ladi-v2 | 3 | 0.000 | 0.000 | n/a | 0.000 | n/a |

(Scores are 0: mockllm emits a fixed sentence. The point is harness proof:
real streamed images → multimodal prompts → routed deterministic scores.)

- FloodNet eval id `dLrmumhnL5jYwpZ2tu4CaD`,
  log `logs/2026-11-09T12-29-59-00-00_floodnet-vqa_hpqCCu8qVJft7FUq9EGwnR.eval`,
  run `runs/footage-floodnet-mock-01/`. Hand-checked sample
  `floodnet-validation-0000-q11` (Complex_Counting, gold `4`,
  image sha `683717e4…`): system + text/image user parts present,
  truth hidden in metadata, `counting_acc=0.0, counting_mae=None`.
- LADI eval log `logs/2026-11-09T12-30-31-00-00_ladi-v2_NWP5f8Hhs57LUZTpZpsxM5.eval`,
  run `runs/footage-ladi-mock-01/`.
- Determinism: two consecutive `build_floodnet_samples(limit=3)` builds
  produce identical ids, questions, and image data URLs.

## Live VLM runs (2026-11-09)

| model | scenario | n | composite | vqa | count | severity | ground |
|---|---|---|---|---|---|---|---|
| muse-spark-open-1026 | floodnet-vqa | 10 | 0.500±0.527 | 0.500±0.535 | 0.500±0.707 | n/a | n/a |
| muse-spark-open-1026 | ladi-v2 | 24 | 0.875±0.338 | 1.000±0.000 | n/a | 0.750±0.452 | n/a |
| Qwen3-VL-8B-Instruct | floodnet-vqa | 10 | 0.300±0.483 | 0.250±0.463 | 0.500±0.707 | n/a | n/a |
| Qwen3-VL-8B-Instruct | ladi-v2 | 24 | 0.917±0.282 | 1.000±0.000 | n/a | 0.833±0.389 | n/a |

Runs: `footage-floodnet-baseten-01` (10 samples, ~8.8k tokens),
`footage-ladi-baseten-01` (24 samples = 2 images × 12 questions, ~31k
tokens), `footage-floodnet-modal-01` (10 samples),
`footage-ladi-modal-01` (24 samples). Manifests under `runs/`, eval
logs under `logs/` (both gitignored per repo policy).

- VLM #1: Baseten `meta-models/muse-spark-open-1026` (scope-locked
  slug). Strong on LADI yes/no (1.000 vqa) but verbose on FloodNet
  counting ("4 roofed structures…" despite the single-word
  instruction), so exact-match counting misses.
- VLM #2: Modal vLLM `Qwen/Qwen3-VL-8B-Instruct` on A10G
  (`infra/modal/vllm_qwen3vl.py`), serverless endpoint
  `https://sanggusti--dmb-vllm-qwen3vl-serve.us-east.modal.run`
  (scales to zero). First deploy failed to start (see
  [inference](inference.md) gotchas); fixed at `--max-model-len 4096`
  + 0.95 utilization. Answers tersely (`3` vs gold `4` on the
  hand-checked sample) — better format compliance, weaker counting.

Repro — VLM #1 (Baseten, scope-locked slug):

```bash
inspect eval evals/floodnet_vqa.py \
  --model openai-api/baseten/meta-models/muse-spark-open-1026
inspect eval evals/ladi_v2.py \
  --model openai-api/baseten/meta-models/muse-spark-open-1026
```

Repro — VLM #2 (Modal vLLM vision; deploy needs an unsandboxed shell;
the inspect provider requires `MODAL_API_KEY` to be set even though
the server checks no key):

```bash
modal deploy infra/modal/vllm_qwen3vl.py  # Qwen3-VL-8B-Instruct
MODAL_API_KEY=dummy MODAL_BASE_URL=<serve-url>/v1 inspect eval evals/floodnet_vqa.py \
  --model openai-api/modal/Qwen/Qwen3-VL-8B-Instruct
```

Record + table (same as the smoke runs):

```bash
python -m hsea.aggregate record --run-id <id> --task floodnet_vqa \
  --model <model-id> --scenario floodnet-vqa --log <run.eval> \
  --scorer-name footage_score
python -m hsea.aggregate table --runs runs --out docs/footage-evals.md \
  --table footage  # then re-add this prose around the table
```

Sample caps: `HSEA_FOOTAGE_LIMIT` (default 25, FloodNet pairs) and
`HSEA_FOOTAGE_MAX_IMAGES` (default 10, LADI images × 12 questions).
