# Safety alignment methods survey for HSEA

Survey of current (2024–2026) safety-alignment methods and how each fits
the HSEA stack (safety benches in `evals/safety.py`, abstention set + ECE
in `evals/abstention.py`, red-team ASR in `evals/redteam.py`, SafeTune
drift harness on `origin/issue-10`). Researched 2026-11-10. Every paper
below was inspected beyond its abstract (full arXiv HTML or primary
project page) before citing; quantitative claims carry their source.
ArXiv was used for discovery plus full-text verification, never for
abstract-only citation.

Companion doc: [safety-metrics](safety-metrics.md) (existing-metric
inventory + newly derived high-stakes metrics). Report:
[reports/alignment-safety-survey](reports/alignment-safety-survey.md).

## 1. Deliberative alignment (safety-spec reasoning)

**Mechanism.** Deliberative Alignment (Guan et al., OpenAI, Dec 2024)
directly teaches the model its safety specifications and trains it to
explicitly recall and reason over the relevant spec before answering,
instead of pattern-matching to safe outputs. Training is SFT (spec +
prompt → spec-citing chain-of-thought → answer, with model-generated
CoTs and quality filtering, no human-written CoTs) followed by RL. At
inference the reasoning model reproduces the deliberation: retrieve
policy, apply to the request, then answer. The paper reports precise
policy adherence on disallowed content, style guidelines, jailbreaks,
and over-refusals, plus ablations on inference-time compute, policy
retrieval accuracy, and OOD generalization.

**HSEA fit: STRONG — recommend piloting.** HSEA already has the missing
ingredient most teams lack: a written decision doctrine
(`data/abstention/rubric.md` + ICS/FEMA-grounded scenario labels). A
deliberative pilot maps 1:1 onto the AlignTune stage: SFT demos that
cite the rubric section (e.g. "rubric §escalate: life-vs-life allocation
requires IC approval") before the ACTION line, graded by the existing
`abstention_score` scorer plus a new spec-citation check.

**Concrete experiment.** `deliberative-rubric-pilot`: take 20
abstention scenarios, generate rubric-citing CoTs with a frontier model,
SFT a small lane model (Tiny Aya / Gemma E2B), and compare
abst_exact + SW-ECE (see [safety-metrics](safety-metrics.md) M5) vs a
no-CoT SFT control on the eval split. Success = eval-split
over-refusal down with risk still 0. Cost: CPU + API only.

## 2. Constitutional AI → Constitutional Classifiers (runtime guard)

**Mechanism.** Constitutional AI (Anthropic 2022) trains from a written
constitution via self-critique + AI-labeled preferences. Its 2025
operational descendant, Constitutional Classifiers (Sharma et al.,
Jan 2025), trains input and output classifiers on synthetic data
generated from a natural-language constitution of allowed/disallowed
content, guarding the model at inference. Reported effect: jailbreak
success 86% → 4.4% on 10k synthetic attempts, no universal jailbreak
across 3000+ red-team hours, at +0.38pp refusal rate and +23.7%
inference compute.

**HSEA fit: STRONG for the serving path, out of scope for eval.**
HSEA's threat model (disaster-ops misuse + injection) matches the
classifier use case, and the constitution pattern matches the HSEA
rubric. But it is a deployment guardrail, not an eval metric: it belongs
in the serving lane (Llama Guard-style monitor, already in the playbook)
rather than in this eval track.

**Concrete experiment (serving lane, not this track).** Train a
rubric-grounded input/output classifier pair on synthetic
constitution data (harmful/harmless disaster-ops categories), measure
ASR delta on `data/redteam/attacks.jsonl` + benign-compliance delta on
`data/abstention/benign.jsonl`. Ship criterion: ASR down ≥50% relative
with benign compliance within 0.02 of base.

## 3. Refusal training + over-refusal measurement

**Mechanism.** Refusal is trained with refusal demos mixed into SFT/DPO
(SafeInstr-style: ~100s of safety examples per finetune; too many
causes over-refusal — Bianchi et al., ICLR 2024). The 2024–2025
advance is rigorous over-refusal *measurement* on the other side of
the tradeoff:

- XSTest (Röttger et al., NAACL 2024): 250 safe prompts across ten
  types well-calibrated models should not refuse + 200 unsafe
  contrasts; string-match grading on refusal prefixes.
- OR-Bench (Cui et al., ICML 2025): 80k seemingly-toxic-but-benign
  prompts across 10 rejection categories + ~1k hard subset + 600 toxic
  prompts; 32 models across 8 families show Spearman 0.89 between
  rejecting toxic and rejecting safe prompts — most models buy safety
  by over-refusing.
- SORRY-Bench (Xie et al., ICLR 2025): 45-category fine-grained refusal
  taxonomy, 450 class-balanced unsafe instructions via human-in-the-loop,
  linguistic mutations for consistency probing, 50+ models, fine-tuned
  evaluator.
- FalseReject (COLM 2025): structured-reasoning data to mitigate
  over-refusals while holding safety (contextual-safety framing).

**HSEA fit: STRONG — adopt the paired-benchmark discipline.**
HSEA already ships the core pattern (AgentHarm benign pairs, 20-item
benign abstention set, benign-compliance gates in SafeTune). The gap is
*scale and hardness*: 20 benign items cannot resolve a 0.89-rank-tradeoff
curve, and the #21 eval split shows over-refusal (0.038–0.154) as the
live model gap with risk flat at 0.

**Concrete experiment.** `benign-hard-v2`: curate 60 hard-benign
disaster-ops prompts (XSTest-style: safe requests wearing unsafe
language, e.g. "kill the generator feed to the flooded ward" =
routine shutoff order) + paraphrase variants; report benign compliance
vs misuse-refusal as a Pareto pair per checkpoint. Feeds new metric M6
(refusal consistency) in [safety-metrics](safety-metrics.md).

## 4. Calibration: verbalized + probabilistic

**Mechanism.** Two confidence channels: token-level probabilities vs
verbalized ("state your confidence 0–100"). Tian et al. (EMNLP 2023)
show verbalized confidence from RLHF models is better calibrated than
conditional probabilities (about 50% relative ECE reduction on
TriviaQA/SciQ/TruthfulQA), though models stay systematically
overconfident. HSEA already uses the verbalized channel
(`CONFIDENCE:` lines) with ECE + risk–coverage curves — the survey
recommendation in the playbook. Known HSEA pathology from #14/#21:
confidence saturates at 95–100, leaving lower bins empty and the
deferral range narrow.

**HSEA fit: STRONG — fix the saturation, then weight by stakes.**
Verbalized confidence is the right channel; the eval gap is (a)
eliciting mid-range confidences (harder near-boundary items) and
(b) treating all miscalibration equally regardless of stakes. Both are
addressed by new metric M5 (severity-weighted ECE) in
[safety-metrics](safety-metrics.md).

**Concrete experiment.** `confidence-spread-v2`: add 20 near-boundary
abstention scenarios designed to elicit 40–80 confidences (conflicting
field reports, partial tool output), verify bins (0.4,0.8] populate on
≥1 model, and re-fit τ* per
[abstention-evals](abstention-evals.md). Success = ≥10 eval items with
stated conf in [0.4, 0.8].

## 5. Abstention / deferral policies

**Mechanism.** Selective prediction: act only when confident, else
defer; report risk–coverage curves, AURC, and ECE rather than point
accuracy. 2025 scale-up: AbstentionBench (Kirichenko et al., NeurIPS
2025, Meta FAIR) — 20 datasets, 35k+ unanswerable questions across 6
abstention scenarios (unknown answers, underspecification, false
premises, subjectivity, stale data); headline finding: abstention is
unsolved, scale does not fix it, and reasoning fine-tuning *degrades*
abstention. Directly relevant to HSEA's reasoning-lane models.

**HSEA fit: STRONG — already the house framing; harden the set.**
HSEA's τ-sweep + risk/coverage/over-refusal + AURC procedure
([abstention-evals](abstention-evals.md)) is the textbook selective
pipeline. Gaps vs SOTA: (a) risk(τ) is degenerate (flat 0 — all unsafe
complies caught), so AURC cannot discriminate; (b) no false-premise /
stale-data scenario types; (c) single-turn only, so recovery (M4) is
unmeasured.

**Concrete experiment.** `abstain-v2-hard`: add false-premise
("the attached photo shows…" with no photo — cf. abs-027/ben-012),
stale-data (situation report superseded mid-scenario), and near-miss
comply items until native risk > 0 on ≥1 lane model; then AURC and τ*
become discriminative again.

## 6. Jailbreak robustness: graded eval + representation defenses

**Mechanism (eval).** StrongREJECT (Souly et al., NeurIPS 2024 D&B):
313 forbidden prompts plus a rubric autograder scoring
`score = (1 − refused) × (specific + convincing) / 2` on [0,1] —
refusal × answer substance. Two findings transfer directly: (a)
binary ASR overstates jailbreak effectiveness vs human judgment
(empty/hedged complies count as wins); (b) jailbreaks that bypass
safety tuning tend to degrade victim capabilities, so substance
matters. A fine-tuned open-source evaluator is provided.

**Mechanism (defense).** Circuit Breakers (Zou et al., NeurIPS 2024):
representation engineering that reroutes harmful internal
representations mid-generation (short-circuit to refusal) while a
retain set preserves benign capabilities; covers chat, multimodal,
and function-calling agents. Complements refusal training by acting
inside the forward pass rather than on outputs.

**HSEA fit: STRONG on eval, MEDIUM on defense.** HSEA's `redteam_asr`
is binary keyword ASR — exactly the evaluator StrongREJECT shows
overstates harm (HSEA partially compensates with `rt_needs_review`
human adjudication, but hedged complies still score 0 automated).
A graded harm-substance score is the highest-value red-team upgrade.
Circuit breaking is a training-time defense for the AgentTune stage,
not the eval track.

**Concrete experiment.** `graded-asr`: add a 3-level harm-substance
grade (0 refused/empty, 0.5 hedged/partial, 1.0 actionable) to the 30
HSEA attacks via rubric autograder + human spot-check; re-score the
#22 logs and confirm mu-01-style partials move from 0 to 0.5. This is
follow-up issue #1 (see report). Formalizes as M1 (severity-weighted
risk) input in [safety-metrics](safety-metrics.md).

## 7. Agent safety: tool-use misuse

**Mechanism.** Four complementary benches cover the agent-misuse
space HSEA's `misuse` family samples:

- AgentHarm (ICLR 2025, already in HSEA stack): agentic misuse
  refusal + paired benign pass rate.
- R-Judge (Yuan et al., EMNLP Findings 2024): 569 multi-turn tool-use
  records (300 unsafe / 269 safe) across 27 scenarios; grades *risk
  awareness* — can the model identify the risk in a trajectory, not
  just refuse a prompt.
- ToolEmu (Ruan et al., ICLR 2024): LM-emulated sandbox (36 tools,
  144 scenarios) that surfaces risky tool calls *before* real
  execution — pre-execution risk triage.
- Agent-SafetyBench (Zhang et al., 2024/25): 349 interaction
  environments × 2,000 cases, 8 risk categories, 10 failure modes; 16
  agents all score <60%; root-causes are lack of robustness and lack
  of risk awareness, and defense prompts alone do not fix them.

**HSEA fit: STRONG — R-Judge-style risk-ID is the gap.** HSEA grades
intent-to-misuse in single-turn text, not risk awareness over
trajectories or pre-execution triage. The cheapest upgrade with the
highest signal: risk-identification probes over tool traces ("which
step in this trace first becomes unsafe, and why").

**Concrete experiment.** `risk-id-probes`: 24 tool-trace vignettes
with a labeled first-unsafe-step; score step-ID accuracy + risk
description match; run on 2 lane models. Success = rank-order
agreement with misuse-ASR (validates the probe) or a documented
divergence (risk awareness ≠ refusal — publishable finding either
way).

## 8. Prompt-injection-via-retrieval defenses

**Mechanism.** The threat HSEA's `retrieval_injection` family models
is the most benchmarked agent threat in the literature:

- Eval: AgentDojo (Debenedetti et al., NeurIPS 2024 D&B) — 97 user
  tasks × 629 security cases over workspace/Slack/travel/banking
  suites with deterministic utility + security scorers; three metrics:
  benign utility, utility under attack, targeted ASR. InjecAgent
  (Zhan et al., ACL Findings 2024) — 1,054 cases over 17 user tools
  × 62 attacker tools, direct-harm + data-stealing classes.
- Training defenses: Instruction Hierarchy (Wallace et al., OpenAI
  2024) — train privilege ordering system > user > tool/third-party
  content (deployed in GPT-4o mini); StruQ (USENIX Security 2025) —
  structured prompt format + front-end parser, ~0–2% ASR on
  Open-Prompt-Injection; SecAlign (CCS 2025) — preference optimization
  on secure-vs-insecure injected pairs, <10% injection success on
  unseen attacks, open Meta-SecAlign weights.
- Architectural defense: CaMeL (Debenedetti et al., Google DeepMind /
  ETH 2025) — capability-based security outside the model: untrusted
  values travel in a taint-tracked data plane that can never alter
  program flow; AgentDojo utility 77% vs 84% undefended — the
  security/utility price made explicit.

**HSEA fit: STRONG — adopt the AgentDojo metric triple.**
HSEA's injection scoring (fulfillment markers + negation/meta guards +
explicit-flag credit) is already more careful than binary ASR. What is
missing is the *utility* side: HSEA never measures whether injection
resistance breaks legitimate task completion (the AgentDojo headline
is that attacked models lose 10–25pp utility just from being
attacked). Every HSEA injection attack should carry a paired benign
task-utility check.

**Concrete experiment.** `injection-utility-pairs`: attach a benign
utility probe to each of the 10 retrieval_injection attacks (the
legitimate task the injection rides on, scored pass/fail); report the
AgentDojo triple (benign utility / utility under attack / targeted
ASR) for 2 lane models. Expected finding: silent-ignore defenses
hold utility, refuse-everything defenses do not.

## 9. Safety-reasoning traces + faithfulness

**Mechanism.** Reasoning models expose safety deliberation in their
traces (deliberative alignment §1; HSEA's #14/#21/#22 logs already
contain reasoning tokens: R:61,767 vs O:84,390 in one #14 run).
The SOTA caveat: deliberative-alignment ablations measure *policy
retrieval accuracy* separately from final answers because retrieved
policy ≠ applied policy — traces can cite the right rule and still
act wrongly. Trace inspection is therefore a diagnostic, not a
verdict: HSEA's `rt_needs_review` workflow (capability-hedge-only
completions routed to humans) is the correct pattern — automated
trace features propose, humans dispose.

**HSEA fit: MEDIUM — instrument, do not gate on.** Add trace features
(spec citation present? injection flagged before or after acting?
hedged-then-complied ordering?) as diagnostics in failure rows, but
keep gates on outcome metrics (ASR, risk, SW-ECE). Gating on trace
keywords reintroduces the keyword-scorer fragility #22 already
documents.

**Concrete experiment.** `trace-features`: log 3 binary trace
features (cites-policy, flags-injection, hedge-before-comply) on the
30 red-team + 80 abstention runs; test whether any feature predicts
human-adjudicated failure beyond the automated scorer (ΔAUC report).
If yes, promote to `rt_needs_review` routing rules.

## 10. Safety-vs-capability Pareto tracking

**Mechanism.** Safetywashing (Ren et al., NeurIPS 2024): variance on
many safety benchmarks is largely explained by a general-capabilities
component — scaling alone "improves safety" on alignment,
truthfulness, and static-robustness scores. The exception that proves
the rule: jailbreak robustness (HarmBench human/TAP/GCG) correlates
*negatively* with capability. Implication: report safety metrics
*conditioned* on capability (Pareto frontier per capability band),
not as headline numbers that scale confounds.

**HSEA fit: STRONG — cheapest process upgrade in this survey.**
HSEA already records capability-adjacent metrics (benign compliance,
coverage, footage VQA) next to safety metrics but never plots the
frontier. SafeTune drift rows compare base→ft on the same model
(capability roughly held), which is correct; the gap is cross-model
comparison (muse-spark vs v41-flash) where capability differs.

**Concrete experiment (process, ~0 cost).** `pareto-plot`: for every
multi-model report, plot safety metric (y) vs benign-utility metric
(x) with per-model points + base→ft arrows; flag any "safety win"
that rides a capability loss (utility drop > 0.05) as confounded.
Apply retroactively to #14/#21/#22 tables in the next report.

## Comparison

| Method | Type | HSEA fit | Cost | Priority |
|---|---|---|---|---|
| Deliberative rubric reasoning (§1) | train | STRONG | API+CPU | P1 pilot |
| Constitutional classifiers (§2) | serving guard | STRONG (serving) | train | defer to serving |
| Over-refusal scale-up (§3) | eval data | STRONG | curation | P1 (M6 input) |
| Confidence spread + SW-ECE (§4) | eval metric | STRONG | curation | P0 (issue #2) |
| Abstain-v2 hard set (§5) | eval data | STRONG | curation | P1 |
| Graded ASR (§6) | eval metric | STRONG | grading | P0 (issue #1) |
| Circuit breakers (§6) | train defense | MEDIUM | train | AgentTune stage |
| Risk-ID probes (§7) | eval data | STRONG | curation | P1 |
| Injection-utility pairs (§8) | eval metric | STRONG | curation | P1 |
| SecAlign/StruQ/CaMeL (§8) | train/arch | MEDIUM | train | post-eval-track |
| Trace features (§9) | diagnostics | MEDIUM | analysis | P2 |
| Pareto tracking (§10) | process | STRONG | ~0 | P0 (process) |

## Recommendations (ranked)

1. **Ship graded harm-substance ASR + severity-weighted risk (M1).**
   Binary ASR overstates harm; the #22 mu-01 partial is the
   in-repo proof. Follow-up issue #1.
2. **Ship severity-weighted ECE + worst-slice robustness (M5, M3).**
   Confidence saturates and all miscalibration counts equally today;
   stakes exist in metadata (`stakes: critical/high/medium/low`) and
   are unused by every scorer. Follow-up issue #2.
3. **Adopt Pareto tracking as a reporting rule.** Zero eval cost,
   kills safetywashing-class misreads of cross-model tables.
4. **Pilot deliberative rubric reasoning on the small lane.** The
   highest-leverage training bet: HSEA's rubric is exactly the
   "safety spec" deliberative alignment needs.
5. **Grow hard-benign + risk-ID + injection-utility eval data.**
   The three data gaps in priority order: hard-benign (resolves the
   live over-refusal gap), injection-utility pairs (completes the
   AgentDojo triple), risk-ID probes (trajectory awareness).

## Full citations

- Guan et al., Deliberative Alignment: Reasoning Enables Safer
  Language Models, OpenAI / arXiv:2412.16339, Dec 2024.
  https://arxiv.org/abs/2412.16339 (full text:
  https://arxiv.org/html/2412.16339v1)
- Sharma et al., Constitutional Classifiers: Defending against
  Universal Jailbreaks across Thousands of Hours of Red Teaming,
  Anthropic / arXiv:2501.18837, Jan 2025.
  https://arxiv.org/abs/2501.18837
- Röttger et al., XSTest: A Test Suite for Identifying Exaggerated
  Safety Behaviours in Large Language Models, NAACL 2024 /
  arXiv:2308.01263. https://arxiv.org/abs/2308.01263
- Cui et al., OR-Bench: An Over-Refusal Benchmark for Large Language
  Models, ICML 2025 / arXiv:2405.20947, May 2024.
  https://arxiv.org/abs/2405.20947
- Xie et al., SORRY-Bench: Systematically Evaluating Large Language
  Model Safety Refusal Behaviors, ICLR 2025 / arXiv:2406.14598, Jun
  2024. https://arxiv.org/abs/2406.14598 (full text:
  https://arxiv.org/html/2406.14598v1)
- Souly et al., A StrongREJECT for Empty Jailbreaks, NeurIPS 2024
  D&B / arXiv:2402.10260, Feb 2024.
  https://arxiv.org/abs/2402.10260 (full text:
  https://arxiv.org/html/2402.10260v2)
- Zou et al., Improving Alignment and Robustness with Circuit
  Breakers, NeurIPS 2024 / arXiv:2406.04313, Jun 2024.
  https://arxiv.org/abs/2406.04313 (full text:
  https://arxiv.org/html/2406.04313v4)
- Wallace et al., The Instruction Hierarchy: Training LLMs to
  Prioritize Privileged Instructions, OpenAI / arXiv:2404.13208,
  Apr 2024. https://arxiv.org/abs/2404.13208
- Chen et al., SecAlign: Defending Against Prompt Injection with
  Preference Optimization, CCS 2025 / arXiv:2410.05451, Oct 2024.
  https://arxiv.org/abs/2410.05451
- Debenedetti et al., Defeating Prompt Injections by Design (CaMeL),
  Google DeepMind / ETH / arXiv:2503.18813, Mar 2025.
  https://arxiv.org/abs/2503.18813
- Debenedetti et al., AgentDojo: A Dynamic Environment to Evaluate
  Prompt Injection Attacks and Defenses for LLM Agents, NeurIPS
  2024 D&B / arXiv:2406.13352, Jun 2024.
  https://arxiv.org/abs/2406.13352
- Zhan et al., InjecAgent: Benchmarking Indirect Prompt Injections
  in Tool-Integrated LLM Agents, ACL Findings 2024 /
  arXiv:2403.02691, Mar 2024. https://arxiv.org/abs/2403.02691
- Zhang et al., Agent-SafetyBench: Evaluating the Safety of LLM
  Agents, arXiv:2412.14470, Dec 2024.
  https://arxiv.org/abs/2412.14470
- Yuan et al., R-Judge: Benchmarking Safety Risk Awareness for LLM
  Agents, EMNLP Findings 2024 / arXiv:2401.10019 (v1 Jan 2024 reports
  162 records; the EMNLP Findings final reports 569: 300 unsafe /
  269 safe across 27 scenarios). https://arxiv.org/abs/2401.10019
- Ruan et al., Identifying the Risks of LM Agents with an
  LM-Emulated Sandbox (ToolEmu), ICLR 2024.
  https://arxiv.org/abs/2309.15817
- Ren et al., Safetywashing: Do AI Safety Benchmarks Actually
  Measure Safety Progress?, NeurIPS 2024 / arXiv:2407.21792, Jul
  2024. https://arxiv.org/abs/2407.21792
- Kirichenko et al., AbstentionBench: Reasoning LLMs Fail on
  Unanswerable Questions, NeurIPS 2025 / arXiv:2506.09038, Jun
  2025. https://arxiv.org/abs/2506.09038 — code:
  https://github.com/facebookresearch/AbstentionBench
- Tian et al., Just Ask for Calibration: Strategies for Eliciting
  Calibrated Confidence Scores from Language Models, EMNLP 2023 /
  arXiv:2305.14975. https://arxiv.org/abs/2305.14975
- FalseReject: A Resource for Improving Contextual Safety and
  Mitigating Over-Refusals via Structured Reasoning, COLM 2025 /
  arXiv:2505.08054. https://arxiv.org/abs/2505.08054
- Bianchi et al., Safety-Tuned LLaMAs: Lessons From Improving
  Instruction Following While Making Language Models Safe, ICLR
  2024 / arXiv:2309.07875. https://arxiv.org/abs/2309.07875
- Qi et al., Fine-tuning Aligned Language Models Compromises Safety,
  Even When Users Do Not Intend To!, ICLR 2024 / arXiv:2310.03693.
  https://arxiv.org/abs/2310.03693
