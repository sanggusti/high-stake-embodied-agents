# Inference — Baseten + Modal + HF through inspect-ai

All providers are OpenAI-compatible, so one routing convention covers all.
Verified against the installed inspect-ai provider: model id
`openai-api/<service>/<model>` reads `<SERVICE>_API_KEY` and
`<SERVICE>_BASE_URL` (upper-cased, `-` → `_`).

## Baseten (paid models)

- Base URL: `https://inference.baseten.co/v1`
- Env: `BASETEN_API_KEY`, `BASETEN_BASE_URL` (see `.env.example`)
- Model id: `openai-api/baseten/<model-slug>` (helper:
  `hsea.models.baseten_model_id`)
- Requirements for mission evals: the slug must support **tool calling**
  (all Model APIs do) and **vision** (model-specific — confirm per slug).
- Status: **verified 2026-10-29** with `meta-models/muse-spark-open-1026`
  (tool calling + vision + reasoning tokens confirmed live).
- Scope lock (permanent): the Baseten key is granted for
  `meta-models/muse-spark-open-1026` ONLY and no further slugs are coming.
  The Baseten column of every sweep is exactly this model; model breadth
  comes from Modal-hosted customs + scripted baselines + human runs, never
  from more Baseten slugs. Do not plan multi-slug matrices.
- Verified: smoke 1.0; forest_fire limit-1 0.608 (24 tool calls, full
  recall, image blocks in transcript); full 6-scenario pack 0.755
  (18/18, ~380k in / 78k out tokens + 1.6M cache reads; $ cost per
  operator's Baseten plan).
- Example: `inspect eval evals/forest_fire.py --model
  openai-api/baseten/meta-models/muse-spark-open-1026`

## Modal (custom models)

- Serve with vLLM in OpenAI-compatible mode (CUDA image, Hugging Face
  + vLLM caches on Modal Volumes). One server file per model:
  `infra/modal/vllm_server.py` (Qwen2.5-7B, hermes parser),
  `infra/modal/vllm_mistral.py` (Mistral-7B-v0.3, mistral parser),
  `infra/modal/vllm_qwen14b.py` (Qwen2.5-14B-AWQ, hermes + `--quantization awq`),
  `infra/modal/vllm_qwen3vl.py` (Qwen3-VL-8B-Instruct, hermes parser,
  A100-80GB, 64K context; issue #3, see below — footage-only A10G config
  retired),
  `infra/modal/vllm_qwen38.py` (Qwen3.8-27B-FP8, qwen3_xml parser,
  A100-80GB, 262K context; issue #1, see below),
  `infra/modal/vllm_molmo2.py` (Molmo2-8B grounding VLM, nightly vLLM +
  custom Hermes tool template, A100-80GB; issue #2, see below),
  `infra/modal/vllm_gemma4.py` (Gemma4-12B unified multimodal, gemma4
  parser, nightly vLLM, A100-80GB, 64K context; issue #3, see below).
- Auth for the CLI: `MODAL_TOKEN_ID` + `MODAL_SECRET_KEY` in `.env`
  (read as `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`), no `modal setup`
  needed.
- After `modal deploy`, set `MODAL_BASE_URL=<serve-url>/v1` and
  `MODAL_API_KEY`.
- Model id: `openai-api/modal/<model-name>` (helper:
  `hsea.models.modal_model_id`).
- Live endpoints (serverless, scale to zero):
  `https://sanggusti--dmb-vllm-openai-compatible-serve.us-east.modal.run`
  (Qwen2.5-7B),
  `https://sanggusti--dmb-vllm-mistral-serve.us-east.modal.run` (Mistral-7B),
  `https://sanggusti--dmb-vllm-qwen14b-serve.us-east.modal.run` (Qwen14B-AWQ),
  `https://sanggusti--dmb-vllm-qwen3vl-serve.us-east.modal.run`
  (Qwen3-VL-8B-Instruct, 2026-11-10; mission config — serverless, stop
  after use),
  `https://sanggusti--dmb-vllm-qwen38-serve.us-east.modal.run`
  (Qwen3.8-27B-FP8, 2026-11-10; serverless, stop after use),
  `https://sanggusti--dmb-vllm-molmo2-serve.us-east.modal.run`
  (Molmo2-8B, 2026-11-10; serverless, stop after use),
  `https://sanggusti--dmb-vllm-gemma4-serve.us-east.modal.run`
  (Gemma4-12B-it, 2026-11-10; serverless, stop after use).
- Status: **verified 2026-10-27** (smoke 1.0, forest_fire limit-1 0.627);
  Mistral + 14B tool-gated 2026-11-04 (forced tool_choice yields valid calls).
- Qwen3.8-27B-FP8: **verified 2026-11-10** (issue #1): slug
  `openai-api/modal/Qwen/Qwen3.8-27B-FP8` (helper:
  `hsea.models.modal_model_id(MODAL_QWEN38_FP8)`); weights ~31 GB FP8
  on one A100-80GB; `--max-model-len 262144` native context;
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml`; smoke +
  forest_fire limit-1 green (see `docs/reports/issue-1-qwen38-serve.md`).
- Molmo2-8B: **verified 2026-11-10** (issue #2): slug
  `openai-api/modal/allenai/Molmo2-8B` (helper:
  `hsea.models.modal_model_id(MODAL_MOLMO2_8B)`); nightly vLLM
  `0.34.0rc1.dev127+g8afb5e9ff` (2026-11-10, pinned by wheel URL);
  weights ~16 GB BF16 on one A100-80GB (A10G OOMs at 16384 ctx);
  `--max-model-len 32768` (near-native 37K: native max is 36864);
  custom Hermes tool template (stock template has no tool/system
  support) + `--tool-call-parser hermes`; smoke 1.0 + forest_fire
  limit-1 0.502 with land+submit loop closure (see
  `docs/reports/issue-2-molmo2-serve.md`). Grounding caveat: Molmo2 is
  the bench's pointing specialist (2/2 FloodNet points dead-on) but a
  weaker autonomous searcher (~0.50 vs 0.61–0.76 baselines; tends to
  confirmation-seek instead of submitting, and hard tool imperatives
  in the template collapse it into refusal) — use it for
  find-it-in-the-frame evals, not solo mission agents.
- Gemma4-12B: **verified 2026-11-10** (issue #3): slug
  `openai-api/modal/google/gemma-4-12B-it` (helper:
  `hsea.models.modal_model_id(MODAL_GEMMA4_12B)`); weights ~24 GB BF16
  on one A100-80GB (47.29 GiB free KV @ util 0.90);
  `--max-model-len 65536` working context (native 256K);
  `--enable-auto-tool-choice --tool-call-parser gemma4` on nightly vLLM
  `0.34.0rc1.dev127+g8afb5e9ff` (stable 0.21.0 predates
  `gemma4_unified`); smoke 1.0 + forest_fire limit-1 0.492 (10 tool
  calls, land+submit; see `docs/reports/issue-3-vlm-workhorses.md`).
- Qwen3-VL-8B: **verified 2026-11-10** (issue #3): slug
  `openai-api/modal/Qwen/Qwen3-VL-8B-Instruct` (helper:
  `hsea.models.modal_model_id(MODAL_QWEN3VL_8B)`); weights ~16 GB BF16
  on one A100-80GB; `--max-model-len 65536`;
  `--enable-auto-tool-choice --tool-call-parser hermes` on stable vLLM
  0.21.0 (NOT qwen3_xml — see gotchas); smoke 1.0 + forest_fire
  limit-1 0.702 (24 tool calls, recall 0.90; see
  `docs/reports/issue-3-vlm-workhorses.md`).
- Gotchas observed: vLLM must be pinned (0.21.0 on CUDA 12.9; 0.32 on
  CUDA 12.4 crashes in FlashInfer JIT); tool calling needs
  `--enable-auto-tool-choice` + the matching `--tool-call-parser`; the sandbox
  blocks Modal's connection (DNS), so Modal commands need an unsandboxed
  shell; Llama repos are HF-gated (401) so the matrix uses ungated
  Mistral/Qwen only; Mistral-7B-v0.3 needs temperature=0 for reliable
  tool calls (temp 1.0 wanders to max-length empties); the retired
  Qwen3-VL-8B footage config on a 24 GB A10G needed `--max-model-len
  4096` + `--gpu-memory-utilization 0.95` (16384 leaves ~0.3 GB KV cache
  and EngineCore fails to start) — mission evals moved it to A100-80GB
  / 64K (issue #3);
  the inspect provider requires `MODAL_API_KEY` set (any value) even
  though the vLLM server checks no key; Qwen3-family models emit XML
  tool calls (`<tool_call><function=name>...`) so they need
  `--tool-call-parser qwen3_xml` (vLLM >= 0.20.2) — `hermes` expects
  JSON and fails silently, leaving calls as plain text (issue #1);
  Qwen3.8's `generation_config.json` overrides sampling to top_k=20 /
  top_p=0.95 (vLLM warns; pass `--generation-config vllm` to ignore);
  first request to a scaled-to-zero app returns HTTP 303 while the
  container cold-starts (~3-4 min for Qwen3.8 incl. 207 s engine init)
  — retry until 200 instead of treating 303 as failure; nightly vLLM
  parses history tool-call `arguments` to dicts before template
  rendering (normalize string-or-dict in custom templates) and forces
  `--disable_chunked_mm_input` for Molmo2's bidirectional mm attention
  (set `--max-num-batched-tokens` to max-model-len); after `modal
  deploy`, verify live behavior — one deploy served the previous
  revision's bytes from a fresh-looking container (issue #2); parser
  choice is per-model, not per-family — Qwen3-VL-8B emits
  Hermes-style calls and needs `hermes` (issue #3), while Qwen3.8 needs
  `qwen3_xml` (issue #1); verify parser choice with a
  `tool_choice="auto"` multi-tool probe, because a forced (named)
  tool_choice masks a mismatch (vLLM constrains forced output to a
  valid call either way); after `modal deploy`, requests can briefly
  keep hitting the previous revision's warm container — verify a
  redeploy on a clean container (`modal app stop` + redeploy) before
  trusting probe results; first request to a scaled-to-zero app hangs
  then 200s on cold-start (~3-7 min incl. weight download) — poll
  `/v1/models` until 200 instead of treating timeouts as failure.

## HF Inference Providers (frontier weights)

- Router base URL: `https://router.huggingface.co/v1` (OpenAI-compatible).
- Env: `HF_API_KEY`, `HF_BASE_URL` (see `.env.example`). `HF_API_KEY` is a
  user access token with Inference Providers permission; any existing
  `HUGGINGFACE_TOKEN` with that permission works as the value — export it
  as `HF_API_KEY` at runtime, never commit it.
- Model id: `openai-api/hf/<hub-repo-id>` (helper:
  `hsea.models.hf_model_id`). Pin a serving backend with
  `hf_model_id(<slug>, provider="<name>")`, which emits the router's
  `<model>:<provider>` syntax (e.g.
  `openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash:deepinfra`).
- Why it exists: the Baseten key is scope-locked to muse-spark-open-1026
  and frontier weights (Command A+, DeepSeek V4.1-Flash, Mistral Large
  3/4, Kimi K2.6) are too big for Modal self-host — this segment is the
  only route for them (see `docs/models.md` §4).
- Status: **verified 2026-11-09** with `deepseek-ai/DeepSeek-V4.1-Flash`
  (smoke 1.0; forest_fire limit-1 0.745, 27 tool calls, full recall).
- Quotas: Free accounts must purchase credits before using Inference
  Providers; PRO/Team/Enterprise get monthly compute credits ($2 list at
  time of writing) applied automatically to routed requests. Quotas and
  rate limits depend on the account tier and the serving provider —
  check the [billing page](https://huggingface.co/settings/billing) and
  [Inference Providers settings](https://huggingface.co/settings/inference-providers/overview)
  (per-model, per-provider usage breakdown) before large sweeps.
- Cost behavior: Hugging Face passes provider rates through with no
  markup; each model+provider pair has its own per-1M-token input/output
  rates, listed on the [router](https://router.huggingface.co/v1/models)
  and the model's provider widgets. Observed 2026-11-09 for
  `deepseek-ai/DeepSeek-V4.1-Flash`: novita/baseten $0.30 in / $1.20 out
  per 1M tokens, deepinfra $0.20 / $0.60. Smoke (~51 tokens) costs a
  fraction of a cent; forest_fire limit-1 (~187k tokens) cost ≈ $0.08
  upper bound at the higher rate. HF spend counts against the same $50
  project cap as Modal (see `docs/reports/modal-spend.md`).
- Billing follows the token (viewer-billed analogue): whoever runs the
  eval pays — each operator exports their own `HF_API_KEY`, so cost
  lands on their account/subscription credits, mirroring the Space
  pattern where the viewer is billed. Team/Enterprise orgs bill a
  selected org. Never share keys.
- Example: `inspect eval evals/forest_fire.py --limit 1 --model
  openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash`

## Verification ladder

1. `inspect eval evals/smoke_inference.py --model mockllm/model` — harness.
2. Same with `--model openai-api/baseten/<slug>` — paid routing + keys.
3. Same with `--model openai-api/modal/<name>` — custom serving.
4. Same with `--model openai-api/hf/<hub-repo-id>` — frontier routing.
5. `evals/forest_fire.py --limit 1` on each real provider — tools + vision.
6. Full sweep (baseline issue) — tracked results table.

Keep `--limit 1` until step 5 passes; provider-backed runs cost money.
