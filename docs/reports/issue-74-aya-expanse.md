# Aya Expanse 8B full loop: SFT + DPO + judge-backed evals (issue #74)

Status: DRAFT (training in flight; result sections fill in on completion).
Objective: full SFT + safety-alignment + judge-backed eval loop for Aya
Expanse 8B, identical to the Qwen3.8 (#8) and Tiny Aya (#73) loops per
[model-comparison-protocol](../model-comparison-protocol.md). No silent
divergence: every deviation is in §6.

Copy rules: no em-dashes; lanes/tracks vocabulary only; every number
traced to a log or manifest; each figure argued in prose.

## 1. Model and data

- Base: `CohereLabs/aya-expanse-8b` @
  `5062468bf9bc0c6035fd64e06274333ec127d980` (verified on the Hub,
  auto-gated, CC-BY-NC-4.0). Text-only, 8K context.
- SFT/align data: `sanggusti/dmb-flywheel-v2` @
  `07c4e27919e8ddabbb358dda3c54ea24a6e7dd66` (local mirror hash-matched
  to the Hub: sft `5b944648...`, dpo `a3224619...`; see
  `data/aya74/manifest.json`).
- Mix (`scripts/compose_aya74.py`, seed 74): flywheel SFT 8,673 +
  HSEA safety demos 78 + abstention demos 80 = 8,831 SFT rows;
  flywheel DPO 7,000 + safety pairs 78 = 7,078 DPO pairs. DisasterM3
  list-content rows joined (750); l2 tool trajectories flattened to
  plain transcripts (251).
- Clearance: no personally-sign source per
  [license-clearance-24](../license-clearance-24.md) (composer asserts).
- Held-out (never trained on): the #19 footage splits (FloodNet-VQA
  validation, LADI-v2 test), `data/redteam/attacks.jsonl` (30),
  `data/dispatch-pairs/dpo.jsonl` (24). Safety (78) and abstention
  (80) rows ARE trained on for contract teaching; those lanes report
  as contract checks, and the report says so wherever they appear.

## 2. Training recipe (shared protocol §2)

- Stage 1 SFT: LoRA r16/alpha32/dropout 0.05, all 7 projections,
  3 epochs, effective batch 8 (2 x 4), LR 2e-4 cosine, warmup 5,
  max_length 4096, packing off, bf16, seed 74.
- Stage 2 alignment: DPO sigmoid beta 0.1 over the merged SFT
  checkpoint (reference = merged SFT, frozen), 1 epoch, effective
  batch 8 (1 x 8), LR 5e-5 cosine, warmup 20, max_length 2048.
- Infra: Modal A100-80GB for SFT+DPO (recipe-identical math; GPU
  class recorded, see §6), A10G merge, L4 vLLM serving.
- Artifacts: adapters + trainer states on the models volume under
  `/models/aya74/`; HF repos §7.

## 3. Before/after tables (protocol §3, metric/base/finetune/delta)

TBD on training completion. Source: `runs/aya74/recompute.json`
(`scripts/aya74_eval.py recompute`, via `scripts/recompute_metrics.py`
`load_samples`), inspect logs `runs/aya74/eval-logs/`.

## 4. Judge lanes T1-T5

TBD. Primary: Meta muse-wonder-1.0 reasoning_effort:low; fallback:
recorded-equivalent HF dual pair. Dual judges, §4 agreement bar
(0.80/kappa 0.60), swap-twice T5 with Conflict Rate, §5 budget rule
($2.00 cap, estimate-first).

## 5. Plots

TBD. Set: deterministic lanes base/sft/dpo; safety Pareto
(refusal vs benign utility); T5 dispatch duels; training loss curves.

## 6. DEVIATIONS from the shared protocol

| # | Area | Protocol | This issue | Reason |
|---|---|---|---|---|
| D1 | SFT max length | same (4096) | 4096, DPO 2048 | DPO dual-model fit on one 80GB GPU; allowed model-specific max length |
| D2 | footage lane | VQA/counting on held-out | text floor with [frame omitted] | Aya Expanse 8B is text-only; cannot take pixels (same class as Tiny Aya) |
| D3 | judges | Meta muse-wonder-1.0 primary | TBD (Meta if key available, else HF dual pair) | no Meta API key in this environment; HF pair is the recorded equivalent |
| D4 | SFT seed | unspecified | 74 | issue seed, recorded |

## 7. Artifacts and spend

- HF: TBD (`sanggusti/aya-expanse-8b-hsea-sft`,
  `sanggusti/aya-expanse-8b-hsea-dpo` with logs + model card).
- Eval ids + fingerprints: TBD (`runs/aya74/recompute.json` trace).
- Sanity: `scripts/recompute_metrics.py --sanity` → SANITY OK
  (footage 4, safety 78, abstention 80, redteam 30).
- Spend: TBD (ledger rows in `docs/reports/modal-spend.md` before PR).
