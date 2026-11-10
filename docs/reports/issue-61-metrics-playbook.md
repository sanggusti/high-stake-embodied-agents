# Issue #61, phase 1: metrics playbook (research only, no re-plots)

Branch: `issue-61-metrics`. Date: 2026-11-10.
Deliverable: [metrics-playbook](../metrics-playbook.md) (decision-complete
spec for the phase 2 re-plot agent) + this report (coverage, top
recommendations, feasibility). No plots redrawn here; no evals run; no PR
opened; #61 stays open for phase 2.

## Coverage

- Local inventory (§1 of the playbook): all four eval families field-mapped
  from direct reads of one sample per report log. Footage logs (11 logs)
  live in the main checkout `logs/`. Safety/abstention/red-team report
  logs (6 logs) live ONLY in the ephemeral issue-14/21/22 worktrees under
  `/tmp`; phase 2 step zero is copying them into the main checkout.
- Paper survey (§9-10 of the playbook): 10 full papers read via ar5iv HTML
  across VLM benchmarks (MMBench, MMMU, POPE), agent/tool-use evals
  (tau-bench, DORA), safety evals (HarmBench, SORRY-Bench, AgentHarm),
  and abstention/calibration (Guo et al., Geifman/El-Yaniv). Three arXiv
  IDs corrected from lookalike physics papers during research and verified
  by fetch (POPE 2305.10355, SORRY-Bench 2406.14598, AgentHarm 2410.09024).
- DORA (§9 of the playbook): full metric set extracted (4 trajectory
  metrics, 4 atomic answer operators + composites, efficiency) with 5
  ranked presentation patterns and a borrow-vs-diverge list. Data stays
  gated per [issue-23-dora-fit](issue-23-dora-fit.md); metrics are public.
- Boring-to-significant mapping (§11): all 7 current plot/table artifacts
  mapped to replacements. Sketches (§12): 6 original ASCII schematics.
  No paper figure copied anywhere.

## Top recommendations per track (one line each)

- Footage: sliced accuracy with Wilson intervals + paired deltas (F1),
  because the 0.500-vs-0.300 gap at n=10 may be noise dressed as fact.
- Serving: cost-per-task table + latency distributions from existing logs
  (S1/S2), the first serving numbers the bench has ever had.
- Safety: Wilson intervals + gate reference lines + paired deltas (SA1),
  plus the AgentHarm-style refusal-vs-benign two-panel (SA2).
- Abstention: risk-coverage with bootstrap bands on zoomed axes (AB1) +
  the coverage-vs-over-refusal panel for the actually-live tradeoff (AB2).
- Red-team: honest ASR intervals ("0/10, consistent with up to 28%") (RT1)
  now, multi-seed pass^k dual curves (RT5) after a $0.16 rerun.
- Flywheel/spend: curation yield funnel (FW1) + rater-agreement bars (FW2)
  + spend burn-down with $/sample (FW3), all from committed files.

Cross-cutting: Wilson intervals on every rate, bootstrap bands on every
curve, paired deltas with tests, a reference line on every plot. None of
the 10 reference papers do this (breadth substitutes for uncertainty at
their scale); at our n it is the whole ballgame.

## Recomputation feasibility table

| Metric | Computable from existing logs? | Missing input | Rerun cost |
|---|---|---|---|
| F1-F4 footage | YES (main `logs/`) | none | $0 |
| F5 grounding | NO | DisasterM3 referring task + runs | <$0.50 |
| S1-S3 serving | YES (all logs + ledger) | none | $0 |
| S4 TTFT/throughput | NO | timing harness on endpoints | <$1, deferrable |
| SA1-SA3 safety | YES (issue-14 worktree logs, copy first) | none | $0 |
| SA4 M2/M3 | YES | none | $0 |
| SA4 M5 on safety stack | NO (no stakes key) | stakes backfill (human ~1h) or abstention-only scope | $0, human time |
| AB1-AB4 abstention | YES (issue-21 worktree logs, copy first) | none | $0 |
| AB5 M1 EHWR | YES but degenerate (risk flat 0) | harder set for signal | $0 |
| AB5 M4 RAUA | NO (no probe turns, zero unsafe episodes) | probe turns on abstain-v2 | <$0.10 |
| RT1-RT4 red-team | YES (issue-22 worktree logs + failures file, copy first) | none | $0 |
| RT5 pass^k | NO (single seed) | 4 more seeds x 30 attacks | ~$0.16 |
| FW1-FW3 flywheel | YES (committed manifests + ledger) | none | $0 |
| FW4 dose curve | NO | SafeTune training runs (#10 follow-up) | training budget |

Verdict: ALL-COMPUTABLE except 6 items, all cheap (combined under $1, all
API class, zero GPU) or human-time-only. The real risk is not cost but
custody: the 6 safety/abstention/red-team report logs exist only in `/tmp`
worktrees. If those vanish before phase 2 copies them, replacement reruns
cost ~$0.29 total (safety $0.18 + abstention $0.07 + redteam $0.04 at
ledgered HF rates) plus re-adjudication of 2 review rows.

## Copy-rule compliance

No em-dashes, lanes/tracks vocabulary, every cited number traces to a
report/ledger/paper section named inline, figures are original sketches,
references are link-with-citation. The #57 no-stacked-figures rule is
encoded as a phase 2 contract item (§0 of the playbook).

## Cost

- Estimate (pre-run): $0. Web research + doc writes only; no inference,
  no Modal. Pre-run ledger TOTAL $6.29, so $6.29 + $0 < $50 → GO.
- Actual: $0.00. Ledger row `61 metrics-playbook cpu 0.5 $0.00`.
- Running TOTAL after this issue phase: **$6.29**.

## Artifacts

- `docs/metrics-playbook.md` (new): the phase 2 spec.
- `docs/reports/issue-61-metrics-playbook.md` (this file).
- `docs/reports/modal-spend.md`: ledger union + this phase's $0.00 row.

## Shutdown

No Modal apps started (explicitly out of scope). Nothing to stop. No GPU.
