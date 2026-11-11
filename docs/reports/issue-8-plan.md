# Issue #8 full-loop execution plan: estimates + decision gates

Pre-GPU estimate-first record (binding). Ledger before this issue:
`docs/reports/modal-spend.md` TOTAL $14.27 against the $100 total cap;
training spend from the ditched overnight attempts is partly
unreported, so this plan targets **≤$32 of the $35 slice** ($3 buffer).
Slice rule: hard ≤$35 Modal; stage 5 (RLVR/GRPO) capped at ~40% (~$14).

## Per-stage pre-run estimates (A100-80GB @ $2.49/hr, slow/fast paths)

Slow = reference impl (~7 s/it SFT, ~13 s/it DPO, ~90 s/it GRPO);
fast = prebuilt fused kernels (~2x throughput). Setup ≈ 30 min/run
(weight load + template render); the SFT probe replaces every anchor
with measured numbers before anything commits.

| stage | steps math | slow $ | fast $ | gate |
|---|---|---|---|---|
| 1 SFT 3ep | 8673/8×3 = 3255 | ~$17.0 | ~$9.4 | probe 50 steps; GO/ABORT at 2x; 3ep iff projected ≤$12 else 2ep (~$11.7/~$6.4) |
| 2 DPO 1ep | 7000/8×1 = 875 | ~$9.1 | ~$5.0 | probe 50 steps; GO/ABORT at 2x |
| 2b IPO arm | same as DPO | ~$9.1 | ~$5.0 | RUN iff tail margin allows; else defer with this estimate (logged) |
| 3 drift audit | dpo-full matrix | ~$3.5 | ~$3.0 | safety+redteam+footage always; missions iff margin allows |
| 4 SafeTune | 459/8×3 = 174 | ~$2.1 | ~$1.5 | SFT-derived estimate (no paid probe); Gate 2 = exit bar |
| 4b Gate-2 re-check | safety-only | ~$0.8 | ~$0.8 | after repair |
| 5 GRPO pilot | 60 steps, G=4 | ~$5.0 | ~$3.5 | probe 10 steps (recorded deviation); 60→30 if slow |
| 5b RFT baseline | 100×8 sample + SFT | ~$2.5 | ~$2.5 | sampling mandatory; filtered-SFT iff margin allows |
| 6 base-full matrix | 1× full battery | ~$3.5 | ~$3.0 | once, reused by every stage |
| 6 final-full matrix | 1× full battery | ~$3.5 | ~$3.0 | on the GRPO checkpoint |
| 6 judges + OOD/BoN | sampled | ~$1.5 | ~$1.5 | hard $2 cap (issue8_judge), OOD/BoN inside serve windows |

Slow-path full plan ≈ $44 (NO-GO) → trims, in order: (1) IPO deferred,
(2) SFT 3→2 epochs, (3) stage-3 audit drops missions (footage-only
benign-flat; missions return at final), (4) GRPO 60→30 steps (1 epoch),
(5) RFT sampling 200→100 prompts; filtered-SFT iff margin allows,
(6) judges sampled to ~$1. Slow+trimmed ≈ $36.7 — still $1.7 over,
so the slow path additionally drops SFT to **1 epoch** (≈$6.4;
LIMA-format argument: 8.6k rows × 1 epoch = 1085 steps is ample for
contract teaching) for ≈**$31.4** total. Fast-path full plan ≈ $33.9
(GO as written, IPO still deferred unless the tail shows margin).

## Gate sequence (each gate re-estimates the TAIL before committing)

1. **SFT probe** → measured s/it, fused-vs-reference path, GPU class.
   Decide SFT epochs (3/2/1) by the table above. ABORT + replan if
   >2x the 7 s/it anchor (>14 s/it).
2. **SFT train** (detached) → record actual $; update tail budget.
3. **DPO probe** → measured s/it; re-estimate tail; lock the trim
   list (missions in stage-3 audit? GRPO 60/30? RFT-SFT?).
4. **DPO train** → record actual $; NO-GO check on the tail.
5. **Base-full + dpo audit** (sequential serves, app stopped after
   each) → Gate 1 verdict via `issue8_drift.py --stage dpo`.
6. **SafeTune train** → safety-only re-check → Gate 2 (`--stage
   safetune`); repair iterates ONLY inside remaining budget.
7. **GRPO probe → pilot** + RFT sampling (+RFT-SFT iff margin).
8. **Final-full + judges + OOD/BoN + replay** → report, ledger, PR.

## Sequencing (#73/#74)

80GB jobs run one at a time across the three issues; every function
carries a fallback chain so a contended class falls through instead
of stalling. #74's finetune was RUNNING when the SFT probe launched;
the probe's chain places it wherever capacity is free, and any
"waiting for GPU" stall kills the launch for a later retry — never
a queue-war.

## Probe log (measured, replaces anchors)

- **Probe 1** (2026-11-11 ~08:50, H200, torch 2.11 cu129, no kernels):
  reference impl, 8.67 s/it, loss 1.157 healthy, 8673 pairs /
  1,980,483 tokens / 43 truncated. Full SFT-3ep extrapolation: 8.34 h
  -> $20.8 A100 / $33 H100 / $37.9 H200. Verdict: time-GO (1.24x)
  but cost-NO-GO on every class. Lesson: the reference path cannot
  fit the slice; the DPO stage (~2x SFT s/it) prices at ~$20 alone.
- **Probe 2** (2026-11-11 ~09:10, H100, torch 2.10 cu128 + prebuilt
  cc1d 1.7.0 + fla 0.6.0): kernels imported (`fused prebuilt`),
  weights loaded, then step-0 crash in bitsandbytes'
  `_dequant_linear_fallback` (M=8192 > 1536 custom-kernel cap ->
  dequant + `F.linear` bf16 -> `CUBLAS_STATUS_INVALID_VALUE`).
  Same fallback trained fine in probe 1 (H200/cu129/torch2.11):
  either a torch-2.10-cu128 cuBLAS regression or H100-specific.
  Isolation repro on L4 decides the fix (torch cu126 build vs
  H100 avoidance). #74's 80GB finetune finished ~09:05, so A100
  placement is available again.
- **Probe 2 postmortem** (no new GPU needed): vLLM v0.17.0 release
  notes document `CUBLAS_STATUS_INVALID_VALUE` on CUDA 12.9+ as a
  CUDA-library-mismatch failure. Probe 2 mixed a 12.9 base image
  with torch's bundled 12.8 (two cuBLAS in one process); probe 1
  matched 12.9/12.9 and trained. Fix: base image CUDA 12.8 to match
  torch cu128 (commit d4d9b5a); serve images stay on 12.9 (vLLM
  validated there). The L4 repro (same mismatched combo) crashed
  too, confirming across architectures.
- **Probe 3** (A100, fused, batch4/accum2/seq2048): 11.52 s/it,
  loss healthy, NO-GO at $27 for SFT-3ep. Fused kernels help ~15%
  (Amdahl: most layers are standard attention + MLP, bnb-bound).
- **Probe 4** (A100, fused, batch1/accum8/seq1536): 14.52 s/it,
  ABORT (2.07x). Micro-batch 1 backfires: 8 accum micro-steps x
  kernel-launch overhead + poor utilization >> custom-kernel win.
  Batch 4 is the sweet spot; no more paid probes after this one
  (in-run gates take over).
- **SFT decision**: 1 epoch (1085 steps x ~11.5 s/it = ~3.5 h,
  ~$9 on A100), launched 11:36 WIB detached with the step-50
  in-run gate armed. LIMA-format argument recorded as the
  DEVIATION (protocol says 3 epochs; slice says 1).
- **DPO gate** (the tail decision point): DPO runs full-data with
  its in-run gate; step-50 measured s/it sets DPO cost and the
  tail budget. Defer order if over: GRPO-training (keep
  RFT-sample + BoN inference-only), dpo-audit missions, RFT size,
  judge sample. DPO-half splitter ready as last resort.
- **Parallelism note**: base-full eval matrix runs DURING SFT
  training (separate A100, separate pool pressure; both placed
  without stalls). Total cost unchanged, ~1.5 h wall saved.

## Deferred-with-estimate log (DEVIATIONS feed)

- DPO on-policy-rejected upgrade (§2.1): ~$5 inference + curation.
- KTO arm (§2.5): DPO-class cost (~$5-9); sanctioned for binary
  abstention labels when pairs do not exist.
- ReMax (§4.3): GRPO-class cost; trigger = GRPO instability only.
- GRPO KL arm (beta>0): needs a ref 27B in memory; ref-free pilot first.
- Cold-start CoT SFT (R1 §3.2 full form): reasoning lives in repair
  RESPONSE texts instead (small-scale pattern).
- vLLM-colocated GRPO generation: does not fit 1×80GB with a 27B
  trainer; standard HF generation used.
