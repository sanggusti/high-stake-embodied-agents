# Issue #6 — HF Inference provider segment (`openai-api/hf/...`)

Branch: `issue-6`. Date: 2026-11-09.

## What

New inspect-ai service segment `openai-api/hf/<hub-repo-id>` in
`hsea/models.py` (`hf_model_id`, `HF_BASE_URL`), following the existing
Baseten/Modal convention. Routes frontier weights that fit neither
Baseten (key scope-locked to muse-spark-open-1026) nor Modal self-host
through the HF Inference Providers router.

## Evals

Model for both: `openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash`
(router default provider routing; no `modal deploy` needed).

| run_id | task | eval id (log) | score | tokens |
|---|---|---|---|---|
| hf-smoke-v41flash | smoke | `logs/2026-11-09T15-16-53-00-00_smoke_Szz6JoogiYrUNKy7goAhMi.eval` | includes 1.000 | 51 (I:45, O:6) |
| hf-ff1-v41flash | forest_fire --limit 1 | `logs/2026-11-09T15-17-03-00-00_forest-fire_9V4WeMzVWrE7Eb8sqwpVVt.eval` | mission 0.745 (recall 1.0, 27 tool calls) | 187,192 (I:16,404, CR:157,056, O:13,732, R:11,648) |

Both recorded via `python -m hsea.aggregate record`
(`runs/hf-smoke-v41flash/`, `runs/hf-ff1-v41flash/`; gitignored local
artifacts). Router reports the model live on novita/baseten/deepinfra
with tool calling + text+image input; the forest_fire transcript
confirms 27 tool calls and full rescue recall end to end.

Duration: smoke ~1s, forest_fire limit-1 1:59 wall.

## Cost

- Estimate (pre-run): smoke ~100 in / ~10 out tokens at the highest
  listed rate ($0.30/$1.20 per 1M) ≈ $0.0001; forest_fire limit-1
  budgeted <$0.15. Pre-run ledger TOTAL was $0.00, so $0.00 + $0.15 <
  $50 cap → GO. Modal GPU estimate $0 (no deploy for this issue).
- Actual: smoke ≈ $0.00002; forest_fire limit-1 ≈ $0.08 upper bound
  (173,460 input+cache tokens × $0.30/1M + 25,380 output+reasoning
  tokens × $1.20/1M; routed provider unknown, so costed at the higher
  novita/baseten rate — deepinfra's $0.20/$0.60 would be ~$0.05).
- HF Inference spend counts against the same $50 project cap as Modal.
- Running TOTAL after this issue: **$0.08**.

## Artifacts

- `hsea/models.py`: `HF_BASE_URL`, `hf_model_id` (+ provider pin).
- `tests/test_models.py`: routing unit tests (hf default + pin).
- `.env.example`: `HF_API_KEY`, `HF_BASE_URL` (names only, no values).
- `docs/inference.md`: HF section (segment, quotas, cost, billing).
- `docs/reports/modal-spend.md`: spend ledger (this issue's rows).

## Shutdown

No Modal apps started (no deploy needed for this issue) — nothing to
stop. Verified: no `modal app stop` required, no GPU spend.
