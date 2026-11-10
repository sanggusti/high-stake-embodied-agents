# Alignment-safety survey: methods + new metrics

Branch: `followup/alignment-safety` (worktree `/tmp/expert-spork-align-safety`,
from `origin/main`). Date: 2026-11-10. Track: HSEA safety alignment +
safety metrics. Spend: $0 (no Modal, no GPU, no model calls — research +
docs + hand-authored SVG only).

## What

Survey of current (2024–2026) AI safety-alignment methods with HSEA fit
verdicts, an inventory of every existing HSEA safety metric with
baselines traced to reports, and six newly derived high-stakes metrics
with formal definitions. Deliverables:

- `docs/alignment-methods-safety.md` — 10-method survey (mechanism,
  HSEA fit verdict, concrete experiment each) + comparison + ranked
  recommendations + full citations (21 sources, all inspected beyond
  abstracts; arXiv used for discovery + full-text verification).
- `docs/safety-metrics.md` — existing-metric inventory (Parts A1–A4
  with baselines from #10/#14/#21/#22) + 6 new metric derivations
  (formula, inputs, range, interpretation, worked micro-example) +
  scorer mapping with prose test cases + baseline backfill.
- `docs/alignment-safety-threat-map.svg` — ORIGINAL threat→defense→
  metric map (committed).
- `docs/alignment-safety-deferral-flow.svg` — ORIGINAL
  calibration/deferral decision flow with new stakes-override branch
  (committed).
- Follow-up issues #46 (graded ASR + consistency) and #47
  (severity-weighted risk + calibration), both `feat, area-eval, p1`.

Figures policy: both figures are original and committed. No published
figures reproduced; external figures are link-with-citation only in
the methods doc. Nothing vendored.

## Methods recommended (ranked, from the survey)

1. Ship graded harm-substance ASR + severity-weighted risk (M1) — #46.
   Binary ASR overstates harm (StrongREJECT); #22 mu-01 is in-repo proof.
2. Ship SW-ECE + worst-slice robustness (M5, M3) — #47. Stakes metadata
   exists and is unused by every scorer.
3. Adopt safety-vs-capability Pareto tracking as a reporting rule
   (Safetywashing) — ~0 cost process change.
4. Pilot deliberative rubric reasoning on the small lane — HSEA's rubric
   is exactly the "safety spec" deliberative alignment needs.
5. Grow hard-benign + risk-ID + injection-utility eval data (XSTest /
   OR-Bench / R-Judge / AgentDojo patterns).

Strong-fit methods deferred out of the eval track: Constitutional
Classifiers (serving guard), Circuit Breakers + SecAlign/StruQ/CaMeL
(training/arch defenses for the AgentTune stage).

## New metrics derived (one line each)

- M1 EHWR (expected-harm-weighted risk): risk weighted by stakes ×
  harm substance — critical misses dominate.
- M2 F_2 escalation: asymmetric-cost escalation score (β=2) with
  safe-direction exclusion — misses cost ~4× false alarms.
- M3 WSR (worst-slice robustness): 1 − max category×stakes slice risk
  — reports the weakest link, not the average.
- M4 RAUA (recovery-after-unsafe-action): P(self-correct within k
  steps | unsafe act), None when unmeasured — second-chance safety.
- M5 SW-ECE (severity-weighted ECE): calibration error weighted by
  stakes — overconfidence where it matters dominates.
- M6 R-CONS (refusal consistency): mean majority agreement across
  paraphrase families + worst-family min — brittle refusal detector.

## Issues created

- [#46](https://github.com/sanggusti/expert-spork/issues/46) —
  Graded harm-substance ASR + refusal consistency for red-team
  harness (M1 grades + M6).
- [#47](https://github.com/sanggusti/expert-spork/issues/47) —
  Severity-weighted risk + calibration-under-stakes for abstention
  eval (M1/M2/M3/M5).

## Baselines traced (inventory sources)

- #14 safety stack: refusal 1.000 both models all refuse benches;
  overall ECE 0.016/0.021; abstain14 escal 0.667/1.000, abstain
  1.000/0.833; agentharm benign 1.000/0.875.
- #21 abstention (eval n=52): risk 0.000, coverage 0.481/0.423,
  over-refusal 0.038/0.154, soft acc 0.981/0.875, ECE 0.036/0.075,
  AURC 0.000, τ* 0.00.
- #22 red-team: v41-flash ASR 0.000 all families + severities
  (automated and adjudicated); injection flag rate 0.400; mockllm
  1.000/0.000/1.000 per family.
- #10 SafeTune (origin/issue-10): tolerances + Gate bars; fixtures
  prove Harden (15×ok) / Steer (ece +0.040) / Recover (7 fail +
  6 severe) all fire.

## Verification

- Full `pytest tests/` green via main-repo `.venv` python (cwd in
  this worktree; main checkout untouched): **244 passed, 7 skipped**
  in 6.39s.
- Both SVGs XML-validated.
- No secrets printed; spend ledger zero row added (union, no rows
  removed).

## Cost

- Estimate: $0 (docs + research only) → GO.
- Actual: $0.00 CPU. No Modal, no GPU, no HF Inference.
- Ledger row: `46 align-safety-survey cpu 0.5 $0.00` (survey
  attributed to first follow-up issue; TOTAL unchanged).

## Shutdown

No Modal apps started — nothing to stop. No GPU spend.
