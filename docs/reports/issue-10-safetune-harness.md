# Issue #10: shared SafeTune safety-drift harness (complete)

Continues the partial branch (`c8b09c8`, synth + mockllm validation,
no checkpoint) to a finished shared harness, consolidating the
proven per-loop components from the merged aya73 (#73) and aya74
(#74) loops. All validation recomputes on RECORDED artifacts on CPU;
GPU spend $0.00.

## What shipped

- `hsea/safety/harness.py` (new): one-call `audit_run_dir` —
  generation loading, SafeTune profiles, base-vs-every-finetune
  drift reports with Harden/Recover/Steer verdicts, Gate 1/2/3
  evaluation, AgentHarm / over-refusal / XSTest panels.
- `hsea/safety/drift.py`: new `evaluate_gates` — formal pass/fail/
  unknown evaluation of the Gate 1/2/3 bars from
  `docs/safety-gates.md` (drift blocks, missing rows yield unknown).
- `hsea/safety/repair.py` (new): shared repair-mix contracts —
  seeded stratified sampling, KTO binary-label validation +
  canonical trainer defaults, unified-footer audit, held-out
  disjointness, manifest builder.
- Delegation (net code deleted): `scripts/aya73_drift.py` /
  `scripts/aya74_drift.py` delegate profiles/panels/rubric to the
  shared harness (loop names kept for compatibility);
  `compose_aya74_repair.stratified` (also used by aya73) delegates
  to `stratified_sample`. Modal KTO legs stay inline (remote code
  cannot import hsea) with parity pinned by test, per the
  reward-mirror precedent.
- CLI: `scripts/safetune_drift.py audit --run-dir ... --checkpoints
  ... --seed ... --out ... --plot` — one command, tables + verdicts
  + gates + plots. Shared plot helpers reused by `plot` and `audit`.
- Docs: `docs/safetune.md` (shared-harness section + rewired
  finetune-workflow section), `docs/safetune-drift-report.md` (the
  drift-report artifact every finetune PR attaches: template +
  filled aya74 base-vs-DPO example with plots).
- Tests: `tests/test_safetune_harness.py` (17 tests). Full suite:
  427 passed, 7 env-skipped.

## Recompute parity (shared vs recorded, CPU)

Every drift report the shared harness produces from the recorded
generations matches the loop-recorded `drift.json` verdict, every
row value/status, and both benign panels exactly:

| Loop | Arms re-audited | Verdicts | Row values | Panels |
|---|---|---|---|---|
| aya74 (`runs/aya74`) | sft, dpo, repair, kto, ipo | 5/5 match (all Recover) | 15/15 rows x 5 arms match | agentharm + overrefusal match |
| aya73 (`runs/aya73-followup`) | sft, dpo, repair, kto, ipo, orpo, rft, grpo | 8/8 match (all Recover) | 15/15 rows x 8 arms match | agentharm + overrefusal (+xstest_safe) match |

Repair-mix regeneration through the shared sampler is byte-identical
to the committed `data/aya74/` mixes (sft/dpo/kto/manifest).

## Verdicts + gates per arm (shared recompute)

| Loop | Arm | Verdict | Fail/severe/drift | Gate 1 | Gate 2 | Gate 3 |
|---|---|---|---|---|---|---|
| aya74 | sft | Recover | 6/2/0 | fail | fail | fail |
| aya74 | dpo | Recover | 7/1/0 | fail | fail | fail |
| aya74 | repair | Recover | 5/1/2 | fail | fail | fail |
| aya74 | kto | Recover | 7/1/0 | fail | fail | fail |
| aya74 | ipo | Recover | 7/1/0 | fail | fail | fail |
| aya73 | sft | Recover | 3/2/0 | fail | fail | pass |
| aya73 | dpo | Recover | 3/4/0 | fail | pass | fail |
| aya73 | repair | Recover | 3/4/0 | fail | fail | pass |
| aya73 | kto | Recover | 3/3/0 | fail | pass | fail |
| aya73 | ipo | Recover | 2/2/0 | fail | pass | pass |
| aya73 | orpo | Recover | 7/5/0 | fail | fail | fail |
| aya73 | rft | Recover | 3/4/0 | fail | pass | fail |
| aya73 | grpo | Recover | 3/4/0 | fail | fail | pass |

Gate 2 stays open on every arm of both loops (no re-grade reaches
Steer): closing it via iterative repair tuning is the follow-on,
which the harness now tracks and expresses but does not perform.

## XSTest lane (both over-refusal definitions)

| Loop | Ckpt | Exact /50 | Hard-refusal /40 | Any-deferral /40 | Controls /10 |
|---|---|---|---|---|---|
| aya74 | base | 47 | 1 | 3 | 10 |
| aya74 | sft | 44 | 5 | 5 | 9 |
| aya74 | dpo | 43 | 5 | 6 | 9 |
| aya74 | repair | 41 | 6 | 8 | 9 |
| aya73 | base | 18 | 10 | 25 | 3 |
| aya73 | sft | 43 | 1 | 7 | 10 |
| aya73 | dpo | 38 | 4 | 12 | 10 |
| aya73 | repair | 43 | 2 | 7 | 10 |
| aya73 | kto | 38 | 4 | 12 | 10 |
| aya73 | ipo | 42 | 1 | 8 | 10 |
| aya73 | orpo | 45 | 4 | 4 | 9 |
| aya73 | rft | 39 | 3 | 11 | 10 |
| aya73 | grpo | 43 | 2 | 7 | 10 |

The shared panel reproduces each loop's recorded `xstest_scores.json`
under its own definition (aya74 hard-refusal, aya73 any-deferral)
and additionally reports the other definition (new columns above:
aya74 any-deferral, aya73 hard-refusal). Note the aya74 creep reads
one refusal worse per checkpoint under deferral semantics (3→8 vs
1→6); controls hold 9–10/10 throughout.

## Repair-mix contracts on the committed mixes

| Check | aya73 | aya74 |
|---|---|---|
| KTO labels valid (des/undes) | 28/45 of 73 ✓ | 80/80 of 160 ✓ |
| Footer audit strict-clean | 28 legacy `ACTION:` (pre-unification, pinned) | 0 leaks, 4-vocab only ✓ |
| Held-out disjoint (redteam/dispatch-pairs/xstest) | ✓ | ✓ |
| KTO defaults mirror Modal legs | ✓ | ✓ |

## Spend

$0.00 GPU/API (CPU recompute only). Ledger row `10 st-shared-audit`
added; file TOTAL unchanged at $56.67 (union baseline per parent:
$91.05 + #8's unreported slice; no GPU ops, no cap risk).

## Out of scope (follow-on)

Closing Gate 2 (iterative repair tuning until a re-grade reaches
Steer) is explicitly out of scope; the harness tracks and expresses
it via verdicts + gates + the repair-mix contract.
