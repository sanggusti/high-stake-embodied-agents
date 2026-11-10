# Metrics playbook: significant metrics + storytelling plots (issue #61, phase 1)

Status: research complete, implementation open (phase 2 re-plots from this doc).
Scope: all six bench tracks (footage, serving, safety, abstention, red-team, flywheel/spend).
Companion report: [issue-61-metrics-playbook](reports/issue-61-metrics-playbook.md).

## 0. How to use this doc (phase 2 contract)

This playbook is decision-complete. The phase 2 agent implements it without
further research: every metric below names its formula, the exact log fields
it computes from, the command or loader that yields those fields, the plot
type with an annotation spec, and the paper reference that justifies it.

Copy rules (from #61, binding on phase 2 output):

- No em-dashes in any prose, label, or annotation. Use commas, colons, or
  hyphens.
- Lanes/tracks vocabulary only: results belong to tracks, methods belong to
  lanes. Never "area", "domain", or "pillar" for either.
- Every number traced: each plotted point cites its source log or manifest
  in the caption or an adjacent trace table.
- No stacked figures without prose (the #57 rule): each figure ships with a
  paragraph that states the claim the figure argues.
- Figures here are original ASCII sketches of what the new plots will look
  like. No paper figure is copied. References are link-with-citation only.

Evidence base: 10 full papers read via ar5iv HTML (never abstract pages
alone), plus the DORA public code paths quoted in
[issue-23-dora-fit](reports/issue-23-dora-fit.md). Full citations in §13.

## 1. Raw-data inventory: what our logs carry

All four inspect-ai eval families log per-sample records with the same
envelope: `metadata` (gold + slice keys), `scores` (one scorer entry with a
`metadata` metric vector), `model_usage` (token counts), `started_at` /
`completed_at` (latency), and `output.completion` (raw text for re-scoring).
Verified by direct read of one sample from each report log on 2026-11-10.

### 1.1 Footage (2 models x 2 corpora; logs present in main checkout `logs/`)

Report logs: `logs/2026-11-09T12-54-34-00-00_floodnet-vqa_*.eval` (muse-spark,
n=10), `logs/2026-11-09T12-55-04-00-00_ladi-v2_*.eval` (n=24), plus the
Modal Qwen3-VL pair and mockllm smoke pair (11 floodnet/ladi logs total).

| Field path | Content | Use |
|---|---|---|
| `metadata.kind` | `counting`, `vqa`, `severity` | Headline slicing |
| `metadata.question_type` | FloodNet type (`Complex_Counting`, ...) | Fine slicing |
| `metadata.answer` | Gold answer text | Re-scoring, POPE-style yes-rate |
| `metadata.image_id`, `metadata.split`, `metadata.revision` | Provenance | Trace audit |
| `scores.footage_score.metadata` | `vqa_accuracy`, `counting_acc`, `counting_mae`, `severity_f1`, `grounding_acc`, `footage_composite` (null when inapplicable) | All rates recomputable |
| `model_usage[*]` | `input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens`, cache reads | Cost-per-task, verbosity analysis |
| `started_at`, `completed_at` | Per-sample wall time | Latency distribution |
| `output.completion` | Raw answer text | Verbosity/format failure taxonomy |

### 1.2 Safety stack (2 models x 78 items; logs in `/tmp/expert-spork-issue-14/logs/`)

Report logs: `logs/2026-11-09T16-35-53-00-00_safety_caDjQESKTMY37FJzB2gsz5.eval`
(muse-spark) and `..._16-36-15-00-00_safety_kJoUXQWapFiCSQ3oEeH2qx.eval`
(DeepSeek-V4.1-Flash). WARNING: these live only in the issue-14 worktree,
which is ephemeral. Phase 2 step zero is copying the two report logs into
the main checkout `logs/` (read-only copy, no re-run).

| Field path | Content | Use |
|---|---|---|
| `metadata.bench`, `metadata.category`, `metadata.label` | Bench (harmbench, airbench, agentharm, safeagent, ...), category, gold label | Per-bench and per-category slices |
| `scores.safety_score.metadata` | `safety_correct`, `refused`, `escalated`, `abstained`, `complied`, `confidence`, `safety_composite`, `label` | All A1 rates + M2 inputs |
| `model_usage`, timestamps, completion | As in §1.1 | Cost, latency, FP/FN audit for M2 |

### 1.3 Abstention (2 models x 80 items; logs in `/tmp/expert-spork-issue-21/logs/`)

Report logs: `logs/2026-11-09T16-23-47-00-00_abstention_njuR2QVGw4VUSJvXPNkLBZ.eval`
(muse-spark) and `..._16-22-40-00-00_abstention_2FX4ZbVYAU4ED6zNZuSz7X.eval`
(v41-flash). Same copy-first warning as §1.2.

| Field path | Content | Use |
|---|---|---|
| `metadata.split`, `metadata.set` | `cal` (n=28) / `eval` (n=52); `risk` / `benign` | Threshold protocol, benign curve |
| `metadata.label`, `metadata.category`, `metadata.stakes` | Gold label, disaster-ops category, `critical`/`high`/`medium`/`low` | M1/M3/M5 inputs, slice curves |
| `scores.abstention_score.metadata` | `abst_exact`, `abst_soft`, `abst_deferred`, `abst_defer_correct`, `abst_confidence`, `abst_pred`, `abst_label` | Curves, ECE, M-slices |
| `model_usage`, timestamps, completion | As in §1.1 | Cost, latency, hedge taxonomy |

### 1.4 Red-team (2 models x 30 attacks; logs in `/tmp/expert-spork-issue-22/logs/`)

Report logs: `logs/2026-11-09T16-50-56-00-00_redteam_Lpa4K4BWN4rHKScmCu9VuM.eval`
(v41-flash) and `..._16-50-54-00-00_redteam_6je68c9FmTxJkKRvZWdFFQ.eval`
(mockllm). Same copy-first warning as §1.2. Adjudication verdicts for the
2 v41 review rows live in `runs/rt-v41flash-seed0/redteam_failures.jsonl`
(issue-22 worktree only; copy alongside).

| Field path | Content | Use |
|---|---|---|
| `metadata.family`, `metadata.variant` | `jailbreak`/`retrieval_injection`/`misuse`; `gcg`/`pair`/`plain`/override-note | Per-family and per-variant ASR |
| `metadata.severity` | `high`/`medium`/`low` | Severity-sliced ASR |
| `metadata.request` | Underlying request text id (e.g. fake evacuation order) | M6 paraphrase-family grouping key |
| `metadata.safetune_probe`, `metadata.markers` | Repair deep-link, fulfillment markers | SafeTune trace, injection audit |
| `scores.redteam_asr.metadata` | `asr`, `rt_refused`, `rt_safety_refusal`, `rt_injection_flagged`, `rt_followed`, `rt_needs_review`, `family`, `severity` | ASR + flag/review rates |
| `model_usage`, timestamps, completion | As in §1.1 | Cost, latency, hedge taxonomy for M1 grades |

### 1.5 Serving (latency + tokens + ledger; no dedicated serving log)

Serving numbers are derived, not logged by a serving harness: per-sample
latency from `completed_at - started_at` on any track log, per-sample
tokens from `model_usage`, GPU cost from
[modal-spend](reports/modal-spend.md) ledger rows. What is NOT in the logs:
TTFT, tokens/sec, queue time, GPU utilization. Those need serving-side
instrumentation (marked NEEDS-RERUN in §4).

### 1.6 Flywheel + spend (manifests + ledger; all committed)

`data/flywheel-v2/manifest.json` (source counts, kept counts, filter notes),
`data/flywheel-v2/*.jsonl` row counts (sft 8,673; dpo 7,000; rlv 250 per
[issue-7-flywheel-v2-curation](reports/issue-7-flywheel-v2-curation.md)),
[modal-spend](reports/modal-spend.md) ledger rows (per-issue run/class/min/$).
All flywheel/spend metrics in §8 compute from these with zero GPU.

## 2. Cross-cutting statistics rules (apply to every track)

Learned the hard way from the survey: none of the 10 reference papers plot
CIs (closest: POPE prompt-template std, SORRY 3-run std, tau-bench pass^k
as variability treatment). Our n is small (10 to 80 per cell), so CIs are
not decoration here, they are the difference between a claim and a rumor.
Phase 2 applies these four rules to every recomputed number:

1. RATES GET WILSON INTERVALS. Every rate (accuracy, ASR, refusal, slice
   rate) reports a 95% Wilson score interval, computed from (successes, n)
   with no new runs. Bars without whiskers are banned.
2. MEANS AND CURVES GET BOOTSTRAP BANDS. Means over samples (composite,
   MAE, latency) report 95% percentile bootstrap CIs (10k resamples, fixed
   seed 61, recorded in the caption). Risk-coverage and reliability curves
   get pointwise bootstrap bands, not just lines.
3. DELTAS ARE PAIRED AND TESTED. Model-vs-model deltas on the same items
   use the paired difference with a bootstrap CI; binary paired outcomes
   add an exact McNemar p-value. Tables carry a delta column with CI, never
   bare side-by-side numbers (SORRY Tables 2/7 style).
4. REFERENCE LINES ON EVERY PLOT. Each plot carries at least one of: a
   chance floor (random/frequent-choice, MMMU style), a weak baseline
   (mockllm, our standing practice), a ceiling (gold/oracle trajectory,
   DORA Table 4 style), or a gate bar (safety-gates.md thresholds).
   Annotate the delta to the reference line on the figure, not just in prose.

## 3. Footage track

Current state: one composite table with std (±0.28 to ±0.71, see
[footage-evals](footage-evals.md)) and no plots at all. Std on n=10 binary
outcomes is nearly uninterpretable; Wilson intervals and slicing carry the
real information.

### F1. Sliced accuracy with Wilson intervals (headline)

- Metric: per-`kind` accuracy (`vqa_accuracy`, `counting_acc`,
  `severity_f1`) plus `footage_composite`, each as rate with 95% Wilson CI.
- Formula: standard rate successes/n; Wilson interval at 95%.
- Rationale: decides whether the muse-spark vs Qwen3-VL gaps (floodnet
  composite 0.500 vs 0.300, ladi 0.875 vs 0.917) are distinguishable from
  noise at n=10/24. Today they read as facts; with n=10 they may not be.
- Computability: COMPUTABLE NOW. `scores.footage_score.metadata` vectors
  in the 4 footage report logs (§1.1). Loader: `read_eval_log` + group by
  `metadata.kind`, or extend `hsea.aggregate table --table footage`.
- Plot: grouped bars (model x kind) with Wilson whiskers, chance-floor
  reference line per kind, annotated paired delta with bootstrap CI.
  MMMU Fig 5 style (grouped bars by slice) + POPE T3 (bold best per block).
- Reference: MMMU headlined micro-averaged accuracy with per-slice
  breakdowns (Yue et al., CVPR 2024, arXiv:2311.16502).

### F2. Accuracy by question type (fine slicing)

- Metric: accuracy per FloodNet `question_type` (Complex_Counting, ...) and
  per LADI damage-grade column family.
- Formula: same rate + Wilson as F1, sliced on `metadata.question_type`.
- Rationale: localizes the failure (MMBench Fig 9 style: "the proprietary
  gap lives in structured inputs"). Decides whether counting or VQA needs
  the next scorer or data investment.
- Computability: COMPUTABLE NOW. `metadata.question_type` is on every
  FloodNet sample (§1.1). Some cells will have n=1; plot them faded with
  full-width intervals rather than dropping them silently.
- Plot: horizontal dot plot with interval whiskers, sorted by gap size,
  n annotated per row. Claim-titled, one slide per corpus.
- Reference: MMBench fine-grained gap analysis (Liu et al., ECCV 2024,
  arXiv:2307.06281).

### F3. Counting MAE distribution + verbosity diagnosis (companion metric)

- Metric: `counting_mae` histogram over counting samples, joined with
  completion length (tokens) and a format-failure flag (first-integer
  missing, per `hsea/scorers/footage.py::extract_int`).
- Formula: MAE is logged; length from `model_usage.output_tokens`;
  flag by re-running `extract_int` on `output.completion`.
- Rationale: decides between two rival explanations of the counting gap:
  perception error (wrong number present) vs instruction-following error
  (verbose "4 roofed structures" vs terse "3"). The baselines differ in
  exactly this way (see [footage-evals](footage-evals.md) VLM notes).
  This is the POPE Yes(%) move: the auxiliary column that embarrasses
  the headline.
- Computability: COMPUTABLE NOW. All fields in §1.1.
- Plot: side-by-side strip plot (MAE vs output tokens per model) with
  format-failure points marked, plus a 2x2 table (correct/format-ok,
  correct/format-bad, wrong/format-ok, wrong/format-bad).
- Reference: POPE diagnostic Yes-ratio companion (Li et al., EMNLP 2023,
  arXiv:2305.10355).

### F4. LADI yes-rate + severity confusion (hallucination lens)

- Metric: yes-rate = P(pred yes) over LADI yes/no items, reported next to
  `vqa_accuracy` and `severity_f1`; plus a 2x2 confusion of predicted vs
  gold damage grade on the 6 severity columns.
- Formula: parse yes/no from `output.completion` (same normalizer as
  `compute_vqa`); gold from `metadata.answer`.
- Rationale: LADI vqa is 1.000 for both models, which is suspicious, not
  triumphant: a 99%-yes model also scores 1.000 on yes-heavy slices.
  Yes-rate decides whether the perfect score reflects perception or bias.
- Computability: COMPUTABLE NOW. Completions + gold answers in §1.1.
- Plot: accuracy bar with a paired yes-rate bar (POPE T3 Yes(%) column
  style) + confusion mini-table with the off-diagonal cells called out.
- Reference: POPE F1 + Yes(%) pairing (Li et al., EMNLP 2023,
  arXiv:2305.10355).

### F5. Grounding accuracy on a referring subset (needs data + rerun)

- Metric: `grounding_acc` (grid-cell match, already scored + unit-tested)
  on a DisasterM3 referring subset, per the plan in
  [footage-evals](footage-evals.md).
- Rationale: grounding is the only footage metric with no measurement at
  all; DORA-style typed scoring (§9) needs a spatial atom, and this is ours.
- Computability: NEEDS RERUN. No corpus task exists. Cost: new task file +
  one API run per model at footage scale (10 to 24 images). Estimate from
  footage ledger rows: under $0.50 total on HF Inference, $0 on Baseten
  scope-locked slug. Not a phase 2 blocker; F1-F4 stand alone.
- Plot: same grouped-bar + Wilson template as F1, added when data exists.
- Reference: DORA point/polygon atoms with distance/IoU gates (Wang et al.,
  arXiv:2605.11633); MMBench spatial-relation slices.

## 4. Serving track

Current state: no serving metrics exist anywhere (no latency/throughput
table in docs/inference.md or docs/models.md). Everything here is new, and
most of it is free from existing logs.

### S1. Cost per eval task (headline efficiency metric)

- Metric: dollars per completed eval sample = provider cost / n, plus
  tokens per sample (input/output split) from `model_usage`.
- Formula: spend from ledger rows matched by run id; tokens summed from
  `model_usage[*].total_tokens` over the log's samples.
- Rationale: decides the provider routing question (Baseten vs Modal vs
  HF Inference) with data instead of vibes. Directly gates the $50 shared
  cap tradeoff for larger n.
- Computability: COMPUTABLE NOW. Ledger + `model_usage` in every log
  (§1.1-1.4). Reports already quote token totals; phase 2 turns them into
  a per-model table with input/output/reasoning splits.
- Plot: horizontal bar table (model x track: $/task, tok/task, reasoning
  share) with the $50-cap burn-down as a reference line. tau-bench cost
  text style ($0.38 agent / $0.23 user-sim per task), promoted to a table.
- Reference: tau-bench per-task cost accounting (Yao et al.,
  arXiv:2406.12045).

### S2. Latency distribution per sample

- Metric: per-sample wall seconds (`completed_at - started_at`), reported
  as median + p90 + max per model x track, with a distribution plot.
- Formula: timestamp diffs; no new instrumentation.
- Rationale: decides whether Modal cold-start/serverless latency dominates
  eval time (the Qwen3-VL A10G endpoint scales to zero; first-sample
  latency vs steady-state is the question).
- Computability: COMPUTABLE NOW from every .eval log (§1.1-1.4).
- Plot: violin or strip plot per model with median/p90 markers, first
  sample flagged (cold-start candidate). DORA scaffold table style: latency
  as a first-class column next to accuracy (their Table 6 carries
  Latency s/task beside AVG).
- Reference: DORA latency column in scaffold ablation (Wang et al.,
  arXiv:2605.11633).

### S3. Verbosity vs score (reasoning-token audit)

- Metric: output + reasoning tokens per sample vs correctness, per model.
- Formula: join `model_usage.reasoning_tokens` / `output_tokens` with the
  per-sample score value.
- Rationale: muse-spark reasoning traces run 1.5k-4.6k tokens per sample
  (see §1.2-1.3 reads); v41-flash runs 1.3k-1.6k. Decides whether the extra
  reasoning buys score or just spend. A flat score-vs-tokens curve is an
  argument to cap reasoning.
- Computability: COMPUTABLE NOW. Both fields in every log.
- Plot: scatter (tokens vs score) with per-model medians, or binned curve.
  Annotate the $/point slope between models.
- Reference: DORA efficiency metric Eff rewards fewer tool calls at fixed
  accuracy (Wang et al., arXiv:2605.11633).

### S4. TTFT and sustained throughput (needs instrumentation)

- Metric: time to first token, tokens/sec steady-state, per deployment.
- Rationale: the real serving SLO pair; decides Modal vs Baseten routing
  for interactive use (Space demos) as opposed to batch evals.
- Computability: NEEDS RERUN with a timing harness (inspect logs carry
  only total latency). Cost: zero-GPU-probe class (short generations on
  already-deployed endpoints), estimate under $1. Phase 2 may defer to a
  serving follow-up; S1-S3 cover the eval-efficiency story without it.
- Plot: TTFT bar + throughput line when data exists.
- Reference: standard serving practice; no paper reference (gap noted in
  §10: none of the 10 papers report serving latency except DORA Table 6).

## 5. Safety track (the #14 stack, 78 items x 2 models)

Current state: per-bench rate table + one grouped-bar PNG
(`issue-14-safety-stack.png`), no CIs, no deltas, no category slicing.

### SA1. Refusal / compliance with Wilson intervals + paired deltas (headline)

- Metric: `refusal_rate`, `benign_compliance`, `safety_correct` per bench,
  each with 95% Wilson CI; model-vs-model paired delta with bootstrap CI
  and McNemar p-value.
- Formula: rates from `hsea/safety/metrics.py` over per-sample
  (`classify_response(completion)`, `metadata.label`) pairs.
- Rationale: per-bench n is 10-16, so one item swings a rate 0.06-0.10
  (the gates doc says this explicitly). The headline question "is
  muse-spark safer than v41-flash" is unanswerable without intervals;
  SA1 makes it answerable or honestly inconclusive.
- Computability: COMPUTABLE NOW. Labels + completions in §1.2 logs.
- Plot: SORRY Table 2 style delta table (base rate + signed delta column)
  plus grouped bars with Wilson whiskers and gate-bar reference lines
  (Gate 1: 0.95, Gate 2: 1.0/0.85, from [safety-gates](safety-gates.md)).
- Reference: SORRY-Bench delta tables (Xie et al., ICLR 2025,
  arXiv:2406.14598); HarmBench ASR standardization (Mazeika et al.,
  ICML 2024, arXiv:2402.04249).

### SA2. Safety-capability two-panel (refusal vs benign, AgentHarm style)

- Metric: paired (harm-bench refusal, benign-bench compliance) per model,
  plotted as the safety-vs-capability tradeoff.
- Formula: same rates as SA1, arranged as harm rows vs benign rows.
- Rationale: a repair that refuses everything is not a repair (Gate 2
  rationale). The two-panel figure separates "refuses more" from "capable
  of less", which is the deployment decision the gates exist to inform.
- Computability: COMPUTABLE NOW. Benign bench rows are in the same logs.
- Plot: AgentHarm Fig 3 style two-panel: top = refusal bars per harm
  bench, bottom = compliance bars on benign, with a convergence annotation
  (ideal: top at 1.0 AND bottom at 1.0). Green benign bars act as the
  capability-ceiling reference.
- Reference: AgentHarm three-panel harm/refusal/non-refusal figure
  (Andriushchenko et al., ICLR 2025, arXiv:2410.09024).

### SA3. Category heatmap (ranked, SORRY style)

- Metric: `safety_correct` per (`bench`, `category`) cell per model.
- Formula: rate over the 2-6 items per cell; Wilson CI in the tooltip or
  appendix table, cell color by rate.
- Rationale: overall 1.000 hides weak categories (the M3 WSR motivation).
  The heatmap is the cheapest worst-slice view: one figure that ranks
  models, slices categories, and carries the story over time as benches
  grow.
- Computability: COMPUTABLE NOW. `metadata.category` on every sample.
- Plot: SORRY-Bench Fig 4 style heatmap: rows = models ranked with
  bracketed overall scores, columns = categories, per-cell rates. Our
  scale is 2 models x ~12 categories, which fits comfortably.
- Reference: SORRY-Bench 56x44 fulfillment heatmap (Xie et al., ICLR 2025,
  arXiv:2406.14598).

### SA4. M2 escalation F2 + M3 WSR + M5 SW-ECE (adopt the new metrics)

- Metric: M2 (asymmetric-cost escalation F2), M3 (worst-slice robustness
  1 - max slice risk), M5 (severity-weighted ECE). Definitions in
  [safety-metrics](safety-metrics.md) Part B.
- Formula: per the safety-metrics micro-examples; FP counts for M2 from
  re-classifying completions; slice keys (category x stakes) for M3;
  stakes weights for M5.
- Rationale: M2 prices the costly error (missed escalation) at 4x the
  false alarm, which is the Gate 3 philosophy made numeric. M3 names the
  weakest slice. M5 answers "calibrated where it matters".
- Computability: COMPUTABLE NOW for M2 (FP from completions) and M3
  (slices from metadata); M5 needs stakes, which exist on abstention logs
  (§1.3) but NOT on safety-stack logs (§1.2 has no stakes key), so M5 on
  the safety stack NEEDS either a stakes backfill (human labels, 78 items,
  ~1 hour) or scope to the abstention track only. Recommend the latter
  for phase 2; backfill is a follow-up.
- Plot: M2 as a single-number card with P/R decomposition bar; M3 as the
  SA3 heatmap with the argmax slice ringed; M5 as a reliability diagram
  with stakes-weighted gaps (Guo style with gap bars, §6).
- Reference: safety-metrics M1-M6 derivations; Guo et al. (ICML 2017,
  arXiv:1706.04599) for the reliability presentation.

## 6. Abstention track (the #21 set, 80 items x 2 models)

Current state: the best-presented track we have (risk-coverage + calibration
PNGs, τ* protocol), but the curves are flat risk-0 lines on wasted axes
(see §11), with no CI bands, no per-slice curves, and no benign-vs-risk
separation in the figures.

### AB1. Risk-coverage with bootstrap bands + honest axes (headline)

- Metric: risk(τ), coverage(τ) on the eval split with pointwise 95%
  bootstrap bands; native point and τ* marked.
- Formula: `hsea/scorers/abstention.py::risk_coverage` over TAU_GRID,
  resampled 10k times (seed 61) for bands.
- Rationale: flat-0 risk at n=52 acted has a wide upper band; the band is
  the honest version of "risk 0.000". Decides whether τ* = 0.00 is a real
  operating point or a small-n artifact. Also rescues the figure: zoom
  axes to the data range instead of 0-1 whitespace.
- Computability: COMPUTABLE NOW. `scripts/abstention_curves.py` already
  loads the items; phase 2 adds the band computation + axis fix.
- Plot: Geifman/El-Yaniv risk-coverage curve style (risk vs coverage,
  per-model lines) with shaded bands, native dots, τ* diamonds, and the
  5% risk gate as a horizontal reference line. Caption records seed + n.
- Reference: Geifman and El-Yaniv risk-coverage curves as the performance
  profile (NeurIPS 2017, arXiv:1705.08500).

### AB2. Coverage-vs-over-refusal tradeoff (the live tradeoff)

- Metric: coverage(τ) vs over-refusal(τ) curve, the tradeoff that is
  actually alive when risk is flat 0 (the abstention-evals doc says this).
- Formula: existing `risk_coverage` outputs; over-refusal among
  comply-labeled items per τ.
- Rationale: decides the real operating question on this set: how much
  benign coverage each model sacrifices (native over-refusal 0.038 vs
  0.154). This is where muse-spark and v41-flash actually differ.
- Computability: COMPUTABLE NOW. Same loader as AB1.
- Plot: second curve panel next to AB1 (same τ grid, shared legend),
  with the native points annotated with their over-refusal values.
- Reference: AgentHarm benign-as-ceiling pattern (Andriushchenko et al.,
  ICLR 2025, arXiv:2410.09024).

### AB3. Reliability diagram with gap bars + count histogram (Guo style)

- Metric: 10-bin reliability (soft-acc vs mean-conf per bin) with gap
  bars, plus a confidence-count histogram on top.
- Formula: `expected_calibration_error` with per-bin weights; counts from
  the same binning.
- Rationale: our current calibration PNG shows lines without gaps or
  counts, so the deepseek 0.8-0.9 bin gap (MCE 0.229) reads as a wiggle.
  Gap bars + counts show it as what it is: overconfidence on a populated
  bin. Decides whether confidence is usable as a defer signal (Gate 3).
- Computability: COMPUTABLE NOW. `abst_confidence` + `abst_soft` in §1.3.
- Plot: Guo Fig 1 style: confidence histogram (top) + reliability diagram
  (bottom) with red gap bars and the diagonal reference. Annotate ECE/MCE
  values on the figure.
- Reference: Guo et al. reliability diagrams + ECE/MCE (ICML 2017,
  arXiv:1706.04599).

### AB4. Per-category and per-stakes slice curves (MMMU style)

- Metric: risk-coverage points and ECE computed per `category`
  (medical-triage, ...) and per `stakes` band.
- Formula: same functions as AB1/AB3, grouped by slice key.
- Rationale: decides WHERE the abstention policy hurts: a 0.154
  over-refusal concentrated in critical-stakes medical items is a
  different ship decision than the same rate spread over low-stakes
  logistics. This is the M3 slice view on the abstention track.
- Computability: COMPUTABLE NOW. Both keys in §1.3 metadata. Small-n
  slices (n<5) report rate + Wilson without curves.
- Plot: small-multiples RC points or a heatmap (slice x model) with the
  worst slice ringed, SORRY Fig 4 style at our scale.
- Reference: MMMU difficulty decomposition table (Yue et al., CVPR 2024,
  arXiv:2311.16502); SORRY-Bench heatmap.

### AB5. M1 EHWR + M4 RAUA status

- Metric: M1 expected-harm-weighted risk at τ*; M4 recovery-after-unsafe
  rate. Definitions in [safety-metrics](safety-metrics.md).
- Rationale: M1 is the stakes-aware headline for the deferral policy;
  M4 measures the second-chance property (seatbelts, not just brakes).
- Computability: M1 COMPUTABLE NOW but degenerate (risk flat 0, so EHWR
  is 0.000 with a vacuous-denominator note; still worth plotting the
  (EHWR, coverage) point to prove the pipeline). M4 NEEDS RERUN: no probe
  turns exist, and with zero unsafe-action episodes the value would be
  None by definition. Cost when abstain-v2 lands: one probe turn per
  unsafe episode, API-cheap (under $0.10 at current per-task rates).
- Plot: M1 as an (EHWR, coverage) point next to the (risk, coverage)
  point; M4 as a card reading "unmeasured, probe protocol ready".
- Reference: safety-metrics Part B; tau-bench multi-trial framing for
  the probe-turn design (Yao et al., arXiv:2406.12045).

## 7. Red-team track (the #22 set, 30 attacks x 2 models)

Current state: ASR-per-family + ASR-by-severity grouped bars, no CIs,
invisible zero bars, no deltas, no failure taxonomy, single seed.

### RT1. ASR with Wilson intervals + paired deltas (headline)

- Metric: ASR per family and overall, automated + adjudicated, each with
  95% Wilson CI; v41-flash vs mockllm paired delta with McNemar p-value.
- Formula: existing `redteam_asr` values; adjudicated ASR flips the 2
  review rows per `redteam_failures.jsonl` verdicts (both attack-failed,
  so adjudicated equals automated here, and the figure should SAY so).
- Rationale: ASR 0.000 at n=10 has Wilson upper bound 0.28: the honest
  headline is "0/10, consistent with true ASR up to 28%", which reframes
  the whole "v41 is robust" story and motivates RT5 multi-seed runs.
- Computability: COMPUTABLE NOW. §1.4 logs + failures file (copy both
  from the issue-22 worktree first).
- Plot: grouped bars with Wilson whiskers (zero bars get a visible marker
  at 0 with the interval printed, never an invisible bar), adjudicated
  vs automated paired bars, annotated delta vs mockllm. HarmBench Fig 5
  style dual panel (model robustness + attack strength).
- Reference: HarmBench ASR headline + greedy-decoding standardization
  (Mazeika et al., ICML 2024, arXiv:2402.04249).

### RT2. ASR by severity and variant (slicing)

- Metric: ASR per severity (high/med/low) and per variant (gcg/pair/plain
  within jailbreak; override/memo/note within injection).
- Formula: same ASR, grouped by `metadata.severity` / `metadata.variant`.
- Rationale: decides which attack style is closest to working. The mu-01
  partial-compliance flip across runs (§8 of redteam.md limitations)
  suggests variant-level differences exist but are invisible at family
  granularity.
- Computability: COMPUTABLE NOW. Both keys in §1.4 metadata. n=5 or less
  per cell: Wilson intervals mandatory, curves banned.
- Plot: small-multiples bars with intervals, or one heatmap (variant x
  severity) with n per cell. SORRY Fig 4 density at our scale.
- Reference: HarmBench per-category bars incl. per-model-class slices
  (Figs 9-11); SORRY-Bench heatmap.

### RT3. Failure taxonomy with counts (DORA Fig 7 style)

- Metric: every non-refused completion classified into a small taxonomy:
  full comply, hedged comply (capability hedge, the rt_needs_review set),
  injection followed, injection flagged, refusal. Counts + share per class.
- Formula: reuse `rt_refused`, `rt_safety_refusal`, `rt_injection_flagged`,
  `rt_followed`, `rt_needs_review` flags already in score metadata; hand
  audit the 2 review rows (done in #22) plus any new non-refusals.
- Rationale: binary ASR cannot score hedged complies (safety-metrics gap
  1, M1 motivation). The taxonomy turns "ASR 0" into "0 complies, 2 hedges
  adjudicated safe, 4 injections flagged of 10", which is a shippable story.
- Computability: COMPUTABLE NOW. All flags in §1.4 score metadata.
- Plot: DORA Fig 7 style taxonomy chart: top-level shares with
  second-level splits and per-class counts. One figure, claim-titled.
- Reference: DORA failure-mode breakdown with percentages (Wang et al.,
  arXiv:2605.11633); MMMU Fig 6 error distribution (35/29/26%).

### RT4. M6 R-CONS over request families (computable now)

- Metric: M6 refusal consistency across paraphrases, grouping by
  `metadata.request` (same underlying request, multiple surface forms).
- Formula: per [safety-metrics](safety-metrics.md) M6: cons_F = majority
  agreement per family, R-CONS = mean, plus worst-family min.
- Rationale: refusal that flips on rewording is a wording detector, not
  a safety property. The GCG/PAIR/plain variant structure is the seed of
  paraphrase families; `request` is the grouping key the playbook needs.
- Computability: COMPUTABLE NOW. `variant` + `request` in §1.4 metadata.
  Caveat: current variants are attack-style variants, not strict
  paraphrases; report as "R-CONS (attack-variant families)" and note the
  stricter paraphrase set as follow-up. Singleton families excluded +
  flagged per the M6 spec.
- Plot: per-family consistency dot plot with the worst family labeled,
  plus the mean card. tau-bench Fig 7 style (per-task sorted curve that
  doubles as an annotation-QA tool: low-consistency families get
  inspected).
- Reference: safety-metrics M6; tau-bench per-task success curves
  (Yao et al., arXiv:2406.12045).

### RT5. Multi-seed pass^k (needs reruns, the tau-bench headline)

- Metric: pass^k = P(all k trials refuse), estimated unbiased from n
  trials per attack; pass@k as the discovery baseline.
- Formula: tau-bench estimators: pass^k = E[C(c,k)/C(n,k)] over attacks
  with c refusals in n trials.
- Rationale: mu-01 flipped across runs at default sampling, so single-seed
  ASR 0 is fragile by direct observation. pass^k turns "flaky" from
  anecdote into the headline visual and informs a reliability gate.
- Computability: NEEDS RERUN. One seed exists; need n>=3 trials per attack
  (5 recommended). Cost: redteam seed0 cost $0.04 on HF Inference
  (ledger row `22 rt-v41flash-seed0`); 4 more seeds ≈ $0.16. Mockllm
  seeds are $0 CPU. Total under $0.25, well inside the cap.
- Plot: tau-bench Fig 4 style dual curves: pass^k solid falling vs
  pass@k dotted rising over k, per family. The single most persuasive
  reliability figure in the survey; steal the layout outright.
- Reference: tau-bench pass^k/pass@k (Yao et al., arXiv:2406.12045).

## 8. Flywheel + spend track

Current state: curation-count tables + one PNG (`issue-7-flywheel-v2-curation.png`),
ledger rows as plain text. No yield, agreement, or spend-efficiency views.

### FW1. Curation yield funnel (headline)

- Metric: kept/source ratio per source (floodnet 1,000 kept, ladi 2,400,
  disasterm3 3,000, crisitext 2,022, humaid 4,000 pairs, crisismmd 3,000
  per the #7 report), plus filter-loss reasons where recorded.
- Formula: counts from `data/flywheel-v2/manifest.json` + the #7 report
  tables; no GPU.
- Rationale: decides where the next curation hour goes: the source with
  the lowest yield per kept-row is the process bottleneck.
- Computability: COMPUTABLE NOW. Manifest + report tables.
- Plot: horizontal funnel bars (sourced → kept per source) with yield %
  annotated, MMMU Fig 3 treemap spirit at table scale.
- Reference: MMMU coverage figure arguing breadth (Yue et al., CVPR 2024,
  arXiv:2311.16502).

### FW2. Label agreement on the spot-check (instrument validation)

- Metric: rater agreement on `data/abstention/rater2.json` and any
  red-team adjudication pairs, as exact-match % and Cohen κ.
- Formula: agreement over the double-labeled items; κ via any stdlib
  implementation (no new dependency: 20 lines).
- Rationale: MMBench Fig 5 / SORRY Table 1 move: validate the ruler
  before measuring with it. Decides whether gold labels are solid enough
  for the WSR and R-CONS slice claims.
- Computability: COMPUTABLE NOW. `rater2.json` is committed; red-team
  adjudication pairs are the 2 review rows (thin but reportable).
- Plot: MMBench Fig 5 style agreement bars (per item subset) with the κ
  value annotated. Opens the metrics section of the next report.
- Reference: MMBench LLM-human alignment bars (91.5%); SORRY-Bench judge
  meta-eval with κ + time cost (Xie et al., ICLR 2025, arXiv:2406.14598).

### FW3. Spend efficiency: $ per eval point + burn-down

- Metric: cumulative ledger spend ($6.29 total at playbook time) broken
  by issue class (api/cpu/a100-80gb/a10g), $ per scored sample per track,
  remaining cap headroom.
- Formula: parse `docs/reports/modal-spend.md` rows; join sample counts
  from report tables.
- Rationale: decides what n the next eval round can afford. The $50 cap
  is shared, so every rerun proposal in this playbook (§3-7 NEEDS RERUN
  items total under $1) must be readable against headroom ($43.71).
- Computability: COMPUTABLE NOW. Ledger is committed text.
- Plot: stacked burn-down (spend over issues, by class) + a $/sample
  table per track. tau-bench cost text promoted to a figure.
- Reference: tau-bench per-task cost accounting (Yao et al.,
  arXiv:2406.12045).

### FW4. Safety dose curve + labeler agreement at scale (future, needs training)

- Metric: gate metrics vs safety-mix fraction (dose curve), per the PT
  note §12.2 `safety_dose_curve` and `labeler_agreement` rows.
- Rationale: decides the safety-data mixture for the next finetune.
- Computability: NEEDS TRAINING RUNS (SafeTune #10 follow-up). Not phase 2.
  The playbook records the plot spec so the training issue inherits it:
  x = safety-mix %, y = gate metric with Wilson bands, one line per gate.
- Plot: dose-response lines with bands; annotate the knee (cheapest mix
  that clears all gates).
- Reference: standard dose-response presentation; HarmBench Table 11
  capability-check pattern (robustness without capability tax).

## 9. DORA metric set in detail (the #61 headline reference)

Paper: Wang et al., "Can LLM Agents Respond to Disasters? Benchmarking
Heterogeneous Geospatial Reasoning in Emergency Operations",
arXiv:2605.11633 (submitted May 2026, v2 Oct 2026). Read in full via ar5iv.
Data stays gated (see [issue-23-dora-fit](reports/issue-23-dora-fit.md));
the metric set below is public and directly reusable.

Benchmark shape: 515 expert-authored tasks over 45 real disaster events
(10 types, 5 continents), 108-tool MCP library in 6 categories, 13 frontier
LLMs evaluated. Task = (query, heterogeneous data manifest, gold tool-call
trajectory T*, structured JSON answer A*). Five dimensions: T1 perception
and assessment, T2 spatial relational analysis, T3 operational planning,
T4 temporal evolution reasoning, T5 multi-modal report synthesis.
Trajectory length grows 3.35 (T1) to 11.96 (T5).

### 9.1 Trajectory metrics (diagnostic, not primary)

Given gold T* and predicted T_pred, DORA reports four measures:

| Metric | Definition |
|---|---|
| Tool-Any-Order | Order-agnostic tool-set recall |
| Tool-In-Order | Longest common subsequence of tool ids, / \|T*\| |
| Tool-Exact-Match | Longest matching prefix length, / \|T*\| |
| Parameter Accuracy | Per-step argument matching, conditioned on correct tool names |

Key empirical pattern: Tool-Any-Order exceeds Tool-Exact-Match by 30+pp
on average (agents know WHICH tools, not the order or arguments). The AP
vs IF ablation (gold tool-order hints) lifts trajectories a lot
(T-Exact-M +26-29pp) but final accuracy only 1.08-4.40pp: the double
bottleneck (tool selection AND argument grounding) is the paper's central
finding.

Borrow for us: our sim-era score vector has no trajectory metric at all
(only outcome scores + tool_calls count). If the mission track ever revives
trajectory grading, Tool-Any/In-Order/Exact-Match + ParamAcc is the
drop-in template, with outcome metrics primary and trajectory diagnostic
(exactly DORA's stated hierarchy: "multiple trajectories can be valid").

### 9.2 Final-answer metrics (primary): four atomic operators

Every typed answer field reduces to one of four atoms (DORA Table 3):

| Operator | Scoring function | Parameter |
|---|---|---|
| scalar | \|yp - yg\| <= tau_r * \|yg\| | tau_r = 0.2 |
| string | nrm(yp) == nrm(yg) | normalized EM |
| point | Euclidean dist <= tau_px | tau_px = 20px |
| polygon | IoU(yp, yg) >= tau_IoU | tau_IoU = 0.5 |

Composites inherit: dict = mean over key-union scalar scores; set = F1
over string-matched elements; line = start point + end point + length.
T5 free summaries are reduced by extracting key statistics + categorical
labels for deterministic scoring. LLM-as-judge + human scoring sit in
Appendix F as backup, never as the headline.

Borrow for us: this is the template for F1/F5 and any future typed eval.
Our footage scorer already mirrors it (normalized EM for VQA, first-int
match for counting, grid-cell match for grounding). Adopt the DORA
tolerances where we lack our own: relative 0.2 for scalar closeness,
IoU 0.5 for any future box/polygon scoring (matches WORKFLOW-adjacent
convention and the M6-style gate philosophy).

### 9.3 Efficiency

Eff = \|T*\| / max(\|T_pred\|, \|T*\|), in (0, 1]. Rewards correct answers
with no redundant tool calls. Headline result: GPT-OSS-120B has the
highest Eff (90.42) with low accuracy: confidently few but wrong tools.
Phase 2 analog: S3 verbosity audit and tool_calls-per-rescue on sim runs.

### 9.4 Presentation patterns to steal (ranked)

1. MAIN TABLE (Table 4): final-answer accuracy per dimension + AVG, then
   trajectory metrics, then efficiency; model groups (baselines /
   commercial / open-source) with a gold-ceiling row on top. Our SA1/F1
   tables should copy the group + ceiling + AVG-first layout.
2. LENGTH-BINNED GAP CURVE (Fig 9): tasks binned by gold-trajectory
   length, per-bucket accuracy for agents vs gold; the gap widens 7% to
   56%. Our analog: score binned by difficulty band (easy/med/hard from
   hsea/difficulty.py) or by τ on RC curves. The caption logic transfers
   verbatim: a rising ceiling + widening gap = agent-side decay, not
   harder tasks.
3. FAILURE TAXONOMY (Fig 7): three disaster-grounding modes with shares
   (damage-semantic 20.3%, sensor-modality 14.8%, pipeline composition
   56.3%) and second-level splits. RT3 copies this shape.
4. ABLATION WITH DELTA ROWS (Tables 5-6): AP vs IF with signed Δ rows;
   scaffold ablation with a latency column. Our RT5 (seed ablations) and
   S2 (latency column) copy both.
5. MODALITY DECOMPOSITION (Fig 8): accuracy by input-modality config
   (pure optical 52-59% vs +OSM 18-27%). Our analog: F2 question-type
   slicing and any future sensor-ablation figure.

### 9.5 Borrow-vs-diverge (for our bench)

Borrow: typed atomic scoring with stated tolerances; trajectory-vs-outcome
hierarchy; gold-ceiling row; length-binned gap curves; failure taxonomy
with shares; latency as a first-class column. Diverge: DORA has no safety
axis (no refusal, abstention, or ASR metrics), gated NC data (design
reference only, never ingested), and no CIs (we add them per §2).

## 10. Paper survey: how the top papers present eval metrics

Each entry: what they headline, how they plot, why it argues significance.
Venues verified against arXiv listing pages; arXiv IDs verified by fetch
(POPE, SORRY-Bench, AgentHarm IDs corrected from lookalike physics papers
during research).

### P1. MMBench (Liu et al., ECCV 2024, arXiv:2307.06281)

Headline: CircularEval top-1 accuracy (MCQ correct only if all circularly
shifted passes agree), overall + per-ability slices; VanillaEval as the
foil that proves CircularEval widens model gaps. Diagnostics: choice-
extraction match rate + LLM-human alignment rate (GPT-4 91.5%).
Plots: radar chart over 20 abilities (hero), choice-bias histograms,
alignment bars, EN-vs-CN grouped bars, fine-grained gap bars, hard-case
grids. Tables: Circular-vs-Vanilla with Acc Change delta column;
mega-leaderboard sorted intra-group with in-house-data flags.
Why: every figure argues a property (discrimination, bias removal,
extractor faithfulness, bilingual robustness, gap localization), never
just a rank. No CIs, no significance tests, cost qualitative only.

### P2. MMMU (Yue et al., CVPR 2024 Oral, arXiv:2311.16502)

Headline: ONE number (micro-averaged accuracy over 11.5K questions);
breadth comes from slicing (6 disciplines, 30 subjects, image types,
Easy/Med/Hard), not from more metrics. Random + Frequent-Choice floors
as reference lines in every table.
Plots: coverage treemap, benchmark positioning table, accuracy-by-image-
type bars, error-distribution chart (perceptual 35%, knowledge 29%,
reasoning 26%), ~100 indexed case figures. Tables: bold best / underline
second-best leaderboard with text-only baselines inline.
Why: the difficulty split (GPT-4V 76/56/31) argues headroom better than
any headline; the error taxonomy converts anecdotes into citable numbers.
No CIs, no human baseline (deferred), no cost/latency.

### P3. POPE (Li et al., EMNLP 2023, arXiv:2305.10355)

Headline: F1 on binary object-probing questions + Yes-ratio diagnostic
(exposes 99%-Yes overconfidence); CHAIR headlined early only to demolish
it (instruction swap doubles scores). Difficulty axis Random → Popular →
Adversarial must decrease monotonically.
Plots/tables: hypothesis bar charts, pipeline schematic, T3 leaderboard
blocked by setting with Acc/P/R/F1/Yes, T4 stability table (Avg±Std across
prompts: POPE 0.78 vs CHAIR 3.22), T5 with F1(Truth) reference column,
T6 POPE-vs-VQA side-by-side (hallucination != VQA skill).
Why: the contrast columns (Δ, ±Std, Truth-ref) turn leaderboards into
arguments about the instrument itself. Only dispersion stat in P1-P3.

### P4. HarmBench (Mazeika et al., ICML 2024, arXiv:2402.04249)

Headline: ASR (attack success rate) with a formal definition, greedy
decoding, T=512 tokens (Fig 2 proves token count shifts ASR up to 30pp:
standardize or compare nothing). Success = exhibited OR clearly tried
(disentangles capability from safety); copyright uses stricter MinHash
matching. Judge: fine-tuned Llama-2-13B distilled from GPT-4, static and
open by design.
Plots: token-sensitivity curve (motivation), dual-panel ASR distributions,
size-vs-family bars, per-category appendix bars. Tables: full model×attack
matrices, capability check (MT-Bench: robustness without capability tax),
judge-agreement table (theirs 93.2 > GPT-4 88.4).
Why: standardization IS the argument; breadth (510 behaviors, 18 attacks,
33 models) substitutes for repeated trials. No CIs (deterministic greedy).

### P5. SORRY-Bench (Xie et al., ICLR 2025, arXiv:2406.14598)

Headline: Fulfillment Rate (lower = safer) from a fine-tuned Mistral-7B
judge; judge meta-metrics are agreement (Cohen κ) + time cost per pass
(explicit accuracy-efficiency frontier justifying the 7B judge).
Plots/tables: giant ranked 56×44 heatmap with bracketed row scores
(signature: ranking + slicing + version tracking in one figure); mutation
tables with signed delta columns (persuasion +5-66pp); settings ablation
with parenthesized deltas; 3-run mean±std (only explicit error analysis
of the safety four).
Why: the heatmap carries model-wise AND category-wise takeaways at once
and supports the values-shift narrative (Llama loosening, Gemini
tightening). Deltas isolate exactly what each variation buys.

### P6. AgentHarm (Andriushchenko et al., ICLR 2025, arXiv:2410.09024)

Headline: THREE numbers always together (harm score with partial credit
via deterministic rubrics, refusal rate via semantic judge on ALL agent
messages, non-refusal harm score as the capability check) + benign-
counterpart scores as the capability ceiling.
Plots: three-panel Fig 3 (harm by attack intensity with green benign bars,
refusal by category, non-refusal scores) with a caption reading rule
(red→green convergence = full capability recovery); score-distribution
histogram proving benchmark separation; execution-log contrasts.
Why: the three panels ARE the argument (attacks work, via refusal
suppression, without degrading capability = genuinely dangerous). No CIs
(T=0 default); no judge-agreement numbers (reliability via deterministic
rubrics + human review of every question).

### P7. tau-bench (Yao et al., arXiv preprint Jun 2024, arXiv:2406.12045)

Headline: pass^1 (binary rule-based reward: exact DB match × output
substrings, zero LLM judging) + pass^k consistency (P(all k trials
succeed), unbiased C(c,k)/C(n,k) estimator) with pass@k as discovery
baseline. gpt-4o <50% overall; pass^8 <25% despite >60% pass^1.
Plots: pass^k solid-falling vs pass@k dotted-rising dual curves
(signature: reliability decay + discovery growth in one frame); grouped
model×method bars; manual failure breakdown (36 failures); difficulty-vs-
writes; per-task sorted success curves (40+ trials, doubles as QA tool);
policy ablation with before→after arrows. Cost in text ($0.38/$0.23/task).
Why: "high average success != deployable reliability" is the core
decision the dual curves force. No plotted CIs (curves are the
variability treatment); no peer-reviewed venue (cite as preprint).

### P8. Guo et al. calibration (ICML 2017, arXiv:1706.04599)

Headline: ECE (weighted mean bin gap, primary) + MCE (worst bin gap, for
high-risk settings); reliability diagrams with gap bars + confidence
histograms as the canonical visual; temperature scaling (single-parameter
Platt) as the recipe.
Why: the diagram + scalar pairing is why ECE stuck: the figure shows
WHERE miscalibration lives, the scalar ranks methods. Our AB3 copies the
pairing exactly; M5 extends the scalar to stakes weights.

### P9. Geifman and El-Yaniv selective classification (NeurIPS 2017, arXiv:1705.08500)

Headline: the risk-coverage curve AS the performance profile ("the entire
performance profile can be specified by its RC curve"); SGR tables mapping
desired risk → train/test risk/coverage + guaranteed bound b* (2% ImageNet
error at ~60% coverage, 99.9% confidence).
Why: the curve replaces point accuracy for any defer-to-human policy; the
SGR table turns a risk target into an operating decision with a guarantee.
Our AB1/AB2 copy the curve; our τ* protocol is the SGR idea without the
formal bound (a documented gap: future work can add exact binomial bounds
on the cal split).

### Cross-cutting gaps (all 10 papers)

No paper plots CIs or runs significance tests on headline numbers; none
weight by severity (SORRY explicitly balances instead); judge-human
agreement appears only in MMBench (91.5%), HarmBench (93.2), SORRY (κ);
serving latency appears only as DORA's Table 6 column; per-task cost
appears only in tau-bench text. Our playbook fills exactly these gaps
(§2 rules), which is how smaller-n bench work earns trust the big
benchmarks get from breadth.

## 11. Boring-to-significant mapping (every current plot)

| Current artifact | What is boring about it | Replace with (this playbook) |
|---|---|---|
| Footage: composite table with ±std, no plots ([footage-evals](footage-evals.md)) | std on n=10 binary outcomes is uninterpretable; no slicing, no deltas | F1 grouped bars + Wilson whiskers + paired deltas; F2 question-type dots; F3 MAE/verbosity strip; F4 yes-rate pair |
| `issue-14-safety-stack.png` (grouped bars, no CIs) | 1.000 bars with no intervals hide the n=10 fragility; no deltas, no gates | SA1 bars + Wilson + gate lines + delta table; SA2 two-panel; SA3 heatmap |
| `issue-21-abstention-risk-coverage.png` | Flat risk-0 lines on 0-1 axes (whitespace); legend soup; no bands; τ* invisible | AB1 zoomed axes + bootstrap bands + 5% gate line; AB2 live-tradeoff panel |
| `issue-21-abstention-calibration.png` | Lines without gap bars or counts; MCE 0.229 reads as a wiggle | AB3 Guo-style histogram + gap-bar diagram with ECE/MCE annotated |
| `issue-22-redteam-asr-family.png` + `-severity` | Invisible zero bars; no CIs; no adjudicated-vs-automated pairing; no deltas | RT1 zero markers + Wilson + adjudicated pairing + delta; RT2 slices; RT3 taxonomy |
| `issue-7-flywheel-v2-curation.png` (counts) | Counts without yield or agreement; spend nowhere | FW1 yield funnel; FW2 agreement bars; FW3 burn-down + $/sample |
| Space Findings tab (tables, no figures) | Numbers without visuals or uncertainty | F1/SA1/RT1 bar figures + AB1 curve, each with the §0 prose paragraph |

## 12. Original figure sketches (what the new plots look like)

ASCII schematics only. Phase 2 renders them in matplotlib (house style:
`matplotlib.use("Agg")`, 150 dpi PNGs, claim titles, traced captions).

### Sketch A: grouped bars with Wilson whiskers + delta annotation (F1/SA1/RT1)

```
accuracy (Wilson 95%)
1.0 |  ###           ***      ### = muse-spark   *** = v41-flash
    |  ###  d=+0.25  ***  - - - - - - Gate 0.95 (SA1) / chance (F1)
0.5 |  #|#  [0.02,   *|*      whiskers = Wilson; bracket = paired-delta CI
    |  ###   0.44]   ***
0.0 +--###-----------***-------------------------
       floodnet      ladi        n=10 / n=24 annotated under each group
```

### Sketch B: risk-coverage with bands + gate line (AB1)

```
risk
0.10| - - - - - - - - - - - - -  5% risk gate
     |  ....muse band....   ____ deepseek band ____
0.00|....................D*====d*___  *=native, D=tau* diamond
     +-------------------------------
     0.40           coverage           0.55   (zoomed x-axis, no whitespace)
```

### Sketch C: reliability diagram with gaps + counts (AB3)

```
count | #             #               confidence histogram (top)
      | #    #        #    #
acc   |                              reliability (bottom), x = mean conf
1.0   |                         [##]  [##] = acc bar, |gap| in red
      |                    |gap|      diagonal y=x reference
0.5   | [##]  [##]  [##]|
      +---------------------------  ECE=0.036 MCE=0.105 annotated
```

### Sketch D: pass^k dual curves (RT5, tau-bench Fig 4 layout)

```
P     1.0 * - - - - - * pass@k (dotted, rising)
      0.5     ___---^^^ pass^k (solid, falling)
      0.0 ___|
          +--------------- k (1..5)    one panel per family
```

### Sketch E: category heatmap with ranked rows (SA3/RT2/AB4)

```
              cat1  cat2  cat3  cat4
muse-spark [0.97]  1.00  1.00 (0.83) 1.00    ring = worst slice (M3 argmax)
v41-flash  [0.88]  1.00 (0.67) 1.00  1.00    [bracket] = row overall
```

### Sketch F: failure taxonomy shares (RT3, DORA Fig 7 layout)

```
refusal 73% |####################  (22/30)
hedged 7%   |##                     (2/30, adjudicated safe)
flagged 13% |####                   (4/10 injections flagged)
followed 0% |                       (0/10)
```

## 13. Full citations

- [DORA] Junjue Wang, Weihao Xuan, Heli Qi, et al. "Can LLM Agents Respond
  to Disasters? Benchmarking Heterogeneous Geospatial Reasoning in Emergency
  Operations." arXiv:2605.11633 (2026). https://arxiv.org/abs/2605.11633
- [MMBench] Yuan Liu, Haodong Duan, Yuanhan Zhang, et al. "MMBench: Is Your
  Multi-modal Model an All-around Player?" ECCV 2024. arXiv:2307.06281.
  https://arxiv.org/abs/2307.06281
- [MMMU] Xiang Yue, Yuansheng Ni, Kai Zhang, et al. "MMMU: A Massive
  Multi-discipline Multimodal Understanding and Reasoning Benchmark for
  Expert AGI." CVPR 2024 (Oral). arXiv:2311.16502.
  https://arxiv.org/abs/2311.16502
- [POPE] Yifan Li, Yifan Du, Kun Zhou, et al. "Evaluating Object
  Hallucination in Large Vision-Language Models." EMNLP 2023.
  arXiv:2305.10355. https://arxiv.org/abs/2305.10355
- [HarmBench] Mantas Mazeika, Long Phan, Xuwang Yin, et al. "HarmBench: A
  Standardized Evaluation Framework for Automated Red Teaming and Robust
  Refusal." ICML 2024. arXiv:2402.04249. https://arxiv.org/abs/2402.04249
- [SORRY-Bench] Tinghao Xie, Xiangyu Qi, Yi Zeng, et al. "SORRY-Bench:
  Systematically Evaluating Large Language Model Safety Refusal."
  ICLR 2025. arXiv:2406.14598. https://arxiv.org/abs/2406.14598
- [AgentHarm] Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian,
  et al. "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents."
  ICLR 2025. arXiv:2410.09024. https://arxiv.org/abs/2410.09024
- [tau-bench] Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan.
  "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World
  Domains." arXiv preprint 2024. arXiv:2406.12045.
  https://arxiv.org/abs/2406.12045
- [Guo-Calibration] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger.
  "On Calibration of Modern Neural Networks." ICML 2017. arXiv:1706.04599.
  https://arxiv.org/abs/1706.04599
- [Selective-Classification] Yonatan Geifman, Ran El-Yaniv. "Selective
  Classification for Deep Neural Networks." NeurIPS 2017. arXiv:1705.08500.
  https://arxiv.org/abs/1705.08500

## 14. Phase 2 work list (ordered, with inputs and outputs)

0. COPY FIRST. Copy the 6 report .eval logs from
   `/tmp/expert-spork-issue-{14,21,22}/logs/` + `runs/rt-*/redteam_failures.jsonl`
   from the issue-22 worktree into the main checkout `logs/`/`runs/`.
   If a worktree is gone, the rerun costs in §14.4 apply instead.
1. STATS HELPERS. One module (suggest `hsea/stats.py`): `wilson(k, n)`,
   `bootstrap_ci(fn, items, seed=61)`, `paired_delta(a, b)`, `mcnemar(a, b)`,
   `cohen_kappa(a, b)`. Unit tests with hand-checked values (spot-check 3
   cells per the #61 test plan).
2. RECOMPUTE (no GPU): F1-F4, S1-S3, SA1-SA4 (M5 abstention-only), AB1-AB4,
   AB5-M1, RT1-RT4, FW1-FW3. Emit one JSON per track with values + CIs +
   trace (log path + sample ids) for the plot-data consistency script.
3. REPLOT: §11 mapping + §12 sketches, matplotlib Agg, 150 dpi, claim
   titles, traced captions, §0 prose paragraph per figure. Before/after
   section in the phase 2 report + /config check + screenshots.
4. CAPPED RERUNS (only with ledger pre-registration, total under $1):
   RT5 (+4 redteam seeds, ~$0.16), F5 (grounding task + runs, <$0.50),
   S4 probes (<$1, deferrable), M4 probes when abstain-v2 lands (<$0.10).
   M5 safety-stakes backfill is human time (~1h), not GPU.
