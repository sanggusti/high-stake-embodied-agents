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
  `infra/modal/vllm_qwen3vl.py` (Qwen3-VL-8B-Instruct vision, no tool parser;
  verified 2026-11-09, see below),
  `infra/modal/vllm_smollm3.py` (SmolLM3-3B text, hermes parser, 64K ctx),
  `infra/modal/vllm_tinyaya.py` (Tiny Aya L2 Thinker 3.35B text, 32K ctx;
  **CC-BY-NC — research-only**, gated HF repo via `hsea-hf` secret),
  `infra/modal/vllm_smolvlm.py` (SmolVLM-256M vision, 8K ctx). All three
  edge-lane servers run on one L4 (issue #4, see below).
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
  (Qwen3-VL-8B-Instruct vision, 2026-11-09).
  Edge lane (2026-11-10, serverless, stop after use):
  `https://sanggusti--dmb-vllm-smollm3-serve.us-east.modal.run`
  (SmolLM3-3B, Apache-2.0),
  `https://sanggusti--dmb-vllm-tinyaya-serve.us-east.modal.run`
  (Tiny Aya L2, CC-BY-NC),
  `https://sanggusti--dmb-vllm-smolvlm-serve.us-east.modal.run`
  (SmolVLM-256M, Apache-2.0).
- Status: **verified 2026-10-27** (smoke 1.0, forest_fire limit-1 0.627);
  Mistral + 14B tool-gated 2026-11-04 (forced tool_choice yields valid calls).
- Edge lane: **verified 2026-11-10** (issue #4): smoke green all three
  (SmolVLM needs `temperature=0` — flaky at default); ff1 floors
  SmolLM3 **64K overflow** (runaway thinking trace), Tiny Aya 0.150,
  SmolVLM 0.150 (both 0 tool calls — templates render no tools; see
  `docs/reports/issue-4-edge-lane.md`). Text models run mission evals
  with `HSEA_VISION=text`.
- **LICENSE — Tiny Aya is CC-BY-NC-4.0 (+ Cohere Labs Acceptable Use
  Policy): research/bench use ONLY. Never ship Tiny Aya outputs,
  weights, or derivatives in a product; flag any pipeline that touches
  it. The other two edge slugs (SmolLM3, SmolVLM) are Apache-2.0.**
- Gotchas observed: vLLM must be pinned (0.21.0 on CUDA 12.9; 0.32 on
  CUDA 12.4 crashes in FlashInfer JIT); tool calling needs
  `--enable-auto-tool-choice` + the matching `--tool-call-parser`; the sandbox
  blocks Modal's connection (DNS), so Modal commands need an unsandboxed
  shell; Llama repos are HF-gated (401) so the matrix uses ungated
  Mistral/Qwen only; Mistral-7B-v0.3 needs temperature=0 for reliable
  tool calls (temp 1.0 wanders to max-length empties); Qwen3-VL-8B on a
  24 GB A10G needs `--max-model-len 4096` + `--gpu-memory-utilization
  0.95` (16384 leaves ~0.3 GB KV cache and EngineCore fails to start);
  the inspect provider requires `MODAL_API_KEY` set (any value) even
  though the vLLM server checks no key; vLLM **rejects**
  `tool_choice="auto"` requests with 400 when the server lacks
  `--enable-auto-tool-choice` + `--tool-call-parser` — set the flags
  even for models whose template renders no tools (calls parse to
  nothing; the episode then measures the no-call floor) (issue #4);
  L4 `web_server` apps scale to zero after ~2 idle minutes and cold
  (re)starts take 6–29 min (vs 3–4 min A100), during which requests
  hang (no fast 303) — chain warm→smoke→ff1 in ONE command with zero
  gap (issue #4); SmolLM3's default thinking mode can run away to
  fill 64K without emitting a tool call (ff1 overflow, issue #4);
  SmolVLM-256M needs `temperature=0` for exact echo (default temp
  paraphrases) and misnames even a solid red square — weakest slug,
  route accordingly.

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
