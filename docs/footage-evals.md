# Footage-grounded VLM eval results (#19)

Tasks: `evals/floodnet_vqa.py` + `evals/ladi_v2.py` — real footage corpora,
no sim. Scorer: `footage_score` (deterministic; value = routed metric,
full vector in metadata — see [metrics](metrics.md)).

## Pinned sources (verified 2026-11-09)

| Corpus | HF id | Revision | Split | License |
|---|---|---|---|---|
| FloodNet-VQA | `ljx620/FloodNet-VQA` | `ec9277af` | validation | untagged on card (annotations derive from FloodNet CDLA-Permissive); eval use only |
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

## Live VLM runs (follow-up — needs provider keys + GPU deploy)

VLM #1 (Baseten, scope-locked slug):

```bash
inspect eval evals/floodnet_vqa.py \
  --model openai-api/baseten/meta-models/muse-spark-open-1026
inspect eval evals/ladi_v2.py \
  --model openai-api/baseten/meta-models/muse-spark-open-1026
```

VLM #2 (Modal vLLM vision; deploy first — needs an unsandboxed shell):

```bash
modal deploy infra/modal/vllm_qwen3vl.py  # Qwen3-VL-8B-Instruct
MODAL_BASE_URL=<serve-url>/v1 inspect eval evals/floodnet_vqa.py \
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
