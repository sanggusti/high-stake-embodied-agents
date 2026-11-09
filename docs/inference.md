# Inference — Baseten + Modal through inspect-ai

Both providers are OpenAI-compatible, so one routing convention covers both.
Verified against the installed inspect-ai provider: model id
`openai-api/<service>/<model>` reads `<SERVICE>_API_KEY` and
`<SERVICE>_BASE_URL` (upper-cased, `-` → `_`).

## Baseten (paid models)

- Base URL: `https://inference.baseten.co/v1`
- Env: `BASETEN_API_KEY`, `BASETEN_BASE_URL` (see `.env.example`)
- Model id: `openai-api/baseten/<model-slug>` (helper:
  `dmb.models.baseten_model_id`)
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
  `infra/modal/vllm_qwen14b.py` (Qwen2.5-14B-AWQ, hermes + `--quantization awq`).
- After `modal deploy`, set `MODAL_BASE_URL=<serve-url>/v1` and
  `MODAL_API_KEY`.
- Model id: `openai-api/modal/<model-name>` (helper:
  `dmb.models.modal_model_id`).
- Live endpoints (2026-11-04, serverless, scale to zero):
  `https://sanggusti--dmb-vllm-openai-compatible-serve.us-east.modal.run`
  (Qwen2.5-7B),
  `https://sanggusti--dmb-vllm-mistral-serve.us-east.modal.run` (Mistral-7B),
  `https://sanggusti--dmb-vllm-qwen14b-serve.us-east.modal.run` (Qwen14B-AWQ).
- Status: **verified 2026-10-27** (smoke 1.0, forest_fire limit-1 0.627);
  Mistral + 14B tool-gated 2026-11-04 (forced tool_choice yields valid calls).
- Gotchas observed: vLLM must be pinned (0.21.0 on CUDA 12.9; 0.32 on
  CUDA 12.4 crashes in FlashInfer JIT); tool calling needs
  `--enable-auto-tool-choice` + the matching `--tool-call-parser`; the sandbox
  blocks Modal's connection (DNS), so Modal commands need an unsandboxed
  shell; Llama repos are HF-gated (401) so the matrix uses ungated
  Mistral/Qwen only; Mistral-7B-v0.3 needs temperature=0 for reliable
  tool calls (temp 1.0 wanders to max-length empties).

## Verification ladder

1. `inspect eval evals/smoke_inference.py --model mockllm/model` — harness.
2. Same with `--model openai-api/baseten/<slug>` — paid routing + keys.
3. Same with `--model openai-api/modal/<name>` — custom serving.
4. `evals/forest_fire.py --limit 1` on each real provider — tools + vision.
5. Full sweep (baseline issue) — tracked results table.

Keep `--limit 1` until step 4 passes; provider-backed runs cost money.
