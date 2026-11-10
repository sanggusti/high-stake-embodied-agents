# HSEA Post-Training / Alignment Methods Survey (2024–2026)

Research notes on CURRENT post-training and alignment methods for LLM
agents, scoped to HSEA: a safety-graded finetuning loop for
disaster-response agents. Stack: CuratorKIT curation, AlignTune SFT,
SafeTune drift repair, AgentTune post-training, Inspect-ai evals.

Grounding: [research-track](research-track.md) (lanes + loop),
[process](process.md) (commands), [alignment-research](alignment-research.md)
(playbook), [models](models.md) (lanes), [safety-gates](safety-gates.md)
(gates). The playbook already plans SFT → DPO/KTO → RLVR/GRPO with
gates; this note surveys what is SOTA in each slot, what to change,
and what to skip — every claim below was checked against the primary
paper HTML/PDF, never the arXiv abs page alone.

Conventions per method: **what** it is, **when it wins**, **what it
costs** (data/compute), **HSEA fit** (adopt/adapt/skip + why), and a
concrete **experiment** on `data/flywheel-v2/` (SFT 8,673 / DPO 7,000
/ RLVR 250 rows) or `data/abstention/` (60 scenarios).

Figures: original diagrams live in
[alignment-methods-posttraining/](alignment-methods-posttraining/)
with provenance in [FIGURES.md](alignment-methods-posttraining/FIGURES.md).
Comparison table: §10. Recommendations: §11. Metrics: §12.
Citations: §13.

## 1. SFT variants

### 1.1 Data-centric SFT (LIMA lesson + Tülu 3 SFT mix)

**What.** Standard next-token finetuning, but the gains come from the
mix, not the optimizer. LIMA shows a 65B model finetuned on only 1,000
carefully curated prompts/responses matches or beats strong baselines
(43% equivalent-or-preferred vs GPT-4 in a controlled human study):
pretraining holds the knowledge, SFT teaches the output contract
[Zhou et al.](#c-lima). Tülu 3's SFT ablations add mix discipline:
diverse chat data, persona data, skill-targeted synthesis, prompt
decontamination, and amount-of-data scaling curves [Lambert et al.](#c-tulu3).

**When it wins.** Small, high-quality demo sets beat large noisy ones;
format/contract adherence (typed dispatch outputs, abstain/escalate
tokens) is exactly what SFT teaches best.

**Cost.** Data: 1k–10k curated demos. Compute: cheapest stage (LoRA
r16, 3 epochs on 1×A10G in the HSEA Modal recipe).

**HSEA fit: ADOPT (already planned; sharpen).** Keep AlignTune SFT
first, but apply LIMA curation (difficulty/diversity/quality filters
as in s1K §6.2) to the 8,673 SFT rows and decontaminate against the
held-out eval sites (DORA anti-leakage discipline).

**Experiment E-SFT.** Curate a 1,000-row "s1K-style" subset of
`sft.jsonl` (stratified by source + difficulty) vs the full 8,673;
finetune Qwen3.8-27B-LoRA on each; grade on footage + dispatch holdout
+ abstention comply-rate. Success: subset within noise of full
(justifies curation-first scaling).

### 1.2 Verifier-filtered SFT (rejection sampling / RFT)

**What.** Sample N completions per prompt, keep only verifier-passing
ones, SFT on those. DeepSeekMath treats RFT as part of the unified
RL paradigm family (RFT/DPO/PPO/GRPO) [Shao et al.](#c-deepseekmath):
it is the offline, verifier-driven cousin of RLVR with no policy
gradient.

**When it wins.** A strong deterministic verifier exists and online RL
is unaffordable; strictly weaker than RL when exploration matters
(Chu et al.: SFT memorizes, RL generalizes — §8.4).

**Cost.** Data: N× inference on the prompt set (N=8–64). Compute:
SFT-cost only.

**HSEA fit: ADOPT as the RLVR baseline.** Every RLVR pilot must beat
RFT on the same verifier, or the RL machinery is unjustified.

**Experiment E-RFT.** For each of the 7,000 dispatch DPO situations,
sample 8 completions from the SFT checkpoint, keep priority-correct
ones, SFT; compare vs §3 RLVR pilot on dispatch holdout + OOD
rule-shifted dispatch (Chu-style: relabeled P1/P2/P3 thresholds).

## 2. DPO family

### 2.1 DPO (+ length-normalized DPO)

**What.** DPO reparameterizes the RLHF reward in closed form and
optimizes a simple classification loss — no reward model, no sampling
during finetuning. It exceeds PPO on sentiment control and matches or
improves summarization/dialogue quality [Rafailov et al.](#c-dpo).
Tülu 3 ablates preference algorithms and standardizes on
**length-normalized DPO** for all final models, with on-policy
preference data (SFT checkpoint's own completions vs others) and
unique (non-SFT-reused) prompts [Lambert et al.](#c-tulu3).

**When it wins.** Paired preference data exists; best cost/quality
default for offline preference tuning.

**Cost.** Data: 1k–100k+ pairs. Compute: needs a frozen reference
model in memory (≈2× policy memory); single-digit GPU hours at 7–27B
with LoRA.

**HSEA fit: ADOPT (already planned; upgrade the recipe).** Switch the
AlignTune slot to length-normalized DPO on on-policy pairs: generate
rejected responses from the HSEA SFT checkpoint itself rather than
reusing only the templated P3-monitor rejections.

**Experiment E-DPO.** See the stage bake-off (§8.5, follow-up issue):
length-norm DPO vs SimPO vs KTO on the 7,000 HumAID/CrisisMMD dispatch
pairs + refusal/abstention pairs; gates: Gate 1 refusal + dispatch
priority accuracy + benign compliance.

### 2.2 IPO (Identity Preference Optimization)

**What.** A ΨPO-framework general objective over pairwise preferences
that bypasses both RLHF approximations (pairwise→pointwise rewards,
reward-model OOD generalization). Setting Ψ to Identity yields an
efficient procedure with performance guarantees; it beats DPO on
illustrative cases where DPO overfits deterministic preferences [Azar
et al.](#c-ipo).

**When it wins.** Small/near-deterministic preference sets where DPO's
unbounded reward gap overfits (e.g. templated chosen/rejected dispatch
texts).

**Cost.** Same data as DPO; same-or-cheaper compute (no ref model
needed in some implementations).

**HSEA fit: ADAPT (ablation arm, not default).** The 7,000 dispatch
pairs are templated enough that DPO overfitting is a live risk; IPO is
the cheap insurance arm in the bake-off.

**Experiment.** Included in E-DPO bake-off as a fourth arm if
AlignTune exposes IPO; else TRL `IPO` trainer on the same pairs.

### 2.3 ORPO (Odds-Ratio Preference Optimization)

**What.** Reference-free **monolithic** objective: NLL + odds-ratio
penalty contrasting favored/disfavored styles in a single SFT stage.
ORPO-tuned Phi-2/Llama-2/Mistral on UltraFeedback alone beat larger
SoTA models (12.20% AlpacaEval 2.0, 66.19% IFEval loose, 7.32
MT-Bench) [Hong et al.](#c-orpo).

**When it wins.** One training stage is all the budget allows;
small models (125M–7B verified) where a separate preference stage
overfits or costs too much.

**Cost.** Data: same pairs as DPO. Compute: ~SFT cost, no ref model.

**HSEA fit: ADAPT (small-lane option).** Not the main Qwen/Gemma path
(two-stage wins there), but the right single-stage recipe for edge
lanes (Tiny Aya 3.35B, Gemma E2B) where the full SFT→DPO chain is
overkill.

**Experiment E-ORPO.** One-stage ORPO vs SFT-only on Tiny Aya using
dispatch pairs; grade dispatch accuracy + abstention + Gate 1. Adopt
for edge lanes if ORPO ≥ SFT on all three.

### 2.4 SimPO (Simple Preference Optimization)

**What.** Reference-free DPO successor: the implicit reward is the
**average log-probability** of the sequence (aligned with generation,
unlike DPO's ref-ratio), plus a target reward margin in the
Bradley-Terry objective. No reference model → more compute/memory
efficient. Beats DPO by up to 6.4 points on AlpacaEval 2 and 7.5 on
Arena-Hard without inflating response length; a Gemma-2-9B-it SimPO
model ranks 1st among <10B models on Chatbot Arena [Meng et
al.](#c-simpo).

**When it wins.** Same regimes as DPO, with lower memory and better
length behavior — strictly the first alternative to try when DPO
underperforms or OOMs.

**Cost.** Data: same pairs as DPO. Compute: ~half of DPO (no ref
model); margin β needs a tiny sweep.

**HSEA fit: ADOPT (co-default with length-norm DPO).** The 27B lane
on 1×80GB GPUs is memory-tight; SimPO's ref-free design fits where
DPO needs offloading. Length control also matters: dispatch outputs
must stay terse, not verbose.

**Experiment.** E-DPO bake-off arm (§8.5).

### 2.5 KTO (Kahneman-Tversky Optimization)

**What.** Prospect-theoretic HALO (human-aware loss) that maximizes
the utility of generations from a **binary** desirable/undesirable
signal — no pairs needed. Matches or exceeds preference-based methods
at 1B–30B despite learning only from binary feedback [Ethayarajh et
al.](#c-kto).

**When it wins.** Only unpaired judgments exist (thumbs up/down on
traces, pass/fail on rollouts); loss-aversion asymmetry fits safety
data (one bad refusal outweighs many good complies).

**Cost.** Data: binary labels (cheaper than pairs; HSEA already has
pass/fail-shaped L2 logs). Compute: DPO-class.

**HSEA fit: ADOPT (unpaired slot; already the playbook fallback).**
Promote from fallback to first-class: watcher L2 trajectories
(`runs/*/trajectories/*.jsonl`) and the 60 abstention scenarios are
naturally binary (correct abstain / unsafe comply), not paired.

**Experiment.** E-DPO bake-off arm: KTO on binarized dispatch outcomes
(priority-correct=good) + abstention labels vs DPO/SimPO on pairs.

## 3. RLVR and verifiable rewards (top recommendation)

### 3.1 RLVR (Tülu 3)

**What.** Standard RL against a **verifiable reward function**: a
constant reward when a completion is successful (checked by
ground-truth outcome, e.g. math answer, IFEval constraint), replacing
learned reward models. RLVR improves GSM8K, MATH, and IFEval in the
final Tülu 3 stage after SFT→DPO [Lambert et al.](#c-tulu3). The
recipe detail that matters: PPO against the verifier with tuned KL
coefficient, run only on skills with ground-truth outcomes.

**When it wins.** Any skill with a deterministic checker: math,
instruction constraints — and, for HSEA, **safety procedures**
(dispatch priority match, abstain-when-labeled, escalate trigger,
confirm-before-irreversible, doctrine citation present).

**Cost.** Data: prompts with checkable answers (HSEA has 7,000
dispatch situations + 60 abstention scenarios + L2 logs — no new
labels needed). Compute: online RL sampling (GRPO-class, §4.2); needs
a fast verifier (HSEA scorers are deterministic Python — ideal).

**HSEA fit: ADOPT — highest-leverage method in this survey.** RLVR is
the missing precise instrument: DPO teaches style, RLVR teaches
checkable safety behavior. It slots into AgentTune exactly where the
playbook already reserves for RLVR/GRPO, and every reward below reuses an
existing scorer or manifest field.

**Experiment E-RLVR (follow-up issue #1).** GRPO on the SFT+DPO
checkpoint with reward = mean of: (a) dispatch priority exact match,
(b) abstain/escalate-label match on abstention prompts, (c)
confirmation-request present before irreversible tool calls in
vignette rollouts, (d) format check (typed output parses). Baseline:
E-RFT on the same verifier. Gates: Gate 3 (SafeAgentBench +
escalation recall + ECE + benign compliance) + dispatch holdout +
OOD rule-shifted dispatch. See
[fig-rlvr-safety.png](alignment-methods-posttraining/fig-rlvr-safety.png).

### 3.2 DeepSeek-R1 rule-based rewards (+ cold start + distill)

**What.** R1-Zero trains reasoning by pure large-scale RL from the
base model with **rule-based** rewards only — accuracy rewards
(deterministic check: boxed answer, compiler/test cases) + format
rewards (`<think>` tags) — explicitly avoiding neural reward models
because of reward hacking and retraining cost. R1 adds multi-stage
training with cold-start data before RL, then RL for all scenarios,
then **distillation** of the reasoning capability into 1.5B–70B dense
models; R1 matches o1-1217 on reasoning [DeepSeek-AI](#c-r1).

**When it wins.** Reasoning behaviors (self-check, backtrack) must
emerge rather than be cloned; neural RMs would be hacked or cost too
much to retrain per stage.

**Cost.** Data: cold-start traces (thousands) + verifiable prompts.
Compute: large-scale RL at frontier scale; the *pattern* (rules, not
learned RM) transfers cheaply to small scale.

**HSEA fit: ADAPT the pattern, not the scale.** Three transfers: (1)
rule-based > neural rewards for the safety-procedure pilot (no RM to
hack, no retrain per gate); (2) cold-start CoT traces for dispatch
reasoning before RL; (3) distill the RLVR-trained 27B reasoner into
edge lanes instead of separately RL-training them.

**Experiment E-R1DISTILL.** After E-RLVR: distill the 27B RLVR
checkpoint's dispatch/abstention reasoning traces (filter:
verifier-passing only) into Gemma E2B via SFT; compare vs E-ORPO
edge recipe on the same evals.

## 4. Online RL: PPO, GRPO, ReMax

### 4.1 PPO

**What.** The classic on-policy algorithm: alternate environment
rollouts with multi-epoch minibatch optimization of a clipped
surrogate objective — TRPO's benefits, simpler, better sample
complexity; the default RLHF optimizer before DPO-era methods
[Schulman et al.](#c-ppo). Needs an actor, a critic (value model), and
a reference model, plus careful hyperparameter tuning.

**When it wins.** General online RL with dense/shaped rewards and
budget for tuning; still the reference implementation behind most RL
recipes (Tülu 3 runs RLVR with PPO and ablates PPO-vs-DPO).

**Cost.** Highest in this survey: 3+ models in memory, rollout
infrastructure, long tuning loops.

**HSEA fit: SKIP as default.** AgentTune lists PPO, but GRPO/ReMax
dominate it on memory and simplicity for the same verifiable-reward
tasks. Keep PPO only as the literature baseline when reporting RLVR
numbers.

**Experiment.** None (covered by E-RLVR's GRPO; cite PPO as the
reference the recipe replaces).

### 4.2 GRPO (Group Relative Policy Optimization)

**What.** PPO variant that **foregoes the critic**: for each question
it samples a group of outputs from the old policy and uses the group's
average reward as the baseline (group-relative advantage), keeping the
clipped objective + KL term. DeepSeekMath-RL 7B gains GSM8K
82.9→88.2% and MATH 46.8→51.7% in-domain plus OOD math gains, with
significantly reduced training resources [Shao et al.](#c-deepseekmath).
R1 adopts GRPO for both R1-Zero and R1 at scale [DeepSeek-AI](#c-r1);
AgentTune ships GRPO.

**When it wins.** Outcome-verifiable tasks where sampling G outputs
per prompt is cheap relative to a critic; the default online-RL
choice for RLVR-style training in 2025–2026.

**Cost.** Data: same verifiable prompts as RLVR. Compute: ~PPO minus
the critic (≈1 model less in memory); rollout cost scales with group
size G (2–64; start G=8 for HSEA pilots).

**HSEA fit: ADOPT as the AgentTune RL optimizer.** Pairs exactly with
§3.1: GRPO + deterministic safety-procedure verifier, no critic to
tune, no neural RM to hack.

**Experiment.** E-RLVR uses GRPO (§3.1); ablate G ∈ {4, 8, 16} on
dispatch reward variance vs wall-time.

### 4.3 ReMax

**What.** REINFORCE-based RLHF exploiting three LLM-task properties
PPO ignores (fast simulation, deterministic transitions,
trajectory-level rewards): no value model, new variance-reduction
technique, 4+ fewer hyperparameters than PPO, ~46% less GPU memory at
7B (fits A800-80GB without offloading), faster training. Applied to
Mistral-7B: 94.78% AlpacaEval win rate, 7.739 MT-Bench — SOTA 7B at
the time [Li et al.](#c-remax).

**When it wins.** Long single-turn generations where GRPO's group
sampling is wasteful (one strong sample + baseline beats G samples);
simplest possible online-RL implementation (advertised as 6 lines over
REINFORCE).

**Cost.** Data: prompts + reward (learned or rule). Compute: below
PPO, comparable to GRPO with small G.

**HSEA fit: ADAPT (fallback if GRPO struggles).** Long tool-trace
rollouts (vignette episodes) make group sampling expensive; ReMax's
single-sample updates fit that regime. Not the default only because
GRPO has the stronger RLVR ecosystem (R1, AgentTune).

**Experiment E-RLALT.** If E-RLVR's GRPO is unstable on multi-step
vignette rollouts, rerun with ReMax on identical rewards/verifier and
compare stability (reward variance) + Gate 3.

## 5. Process supervision and PRMs

### 5.1 PRMs (Let's Verify Step by Step)

**What.** **Process supervision** (feedback per reasoning step) vs
outcome supervision (feedback on the final answer). Process
supervision wins significantly on MATH: 78% on a representative MATH
subset; active learning further improves label efficiency. Releases
PRM800K (800k step-level human labels) [Lightman et
al.](#c-prm). R1's authors notably report PRMs as an *unsuccessful
attempt* at their scale (reward hacking, retraining cost) — use rules
where checkable, PRMs where not [DeepSeek-AI](#c-r1).

**When it wins.** Multi-step reasoning with expensive final outcomes;
step labels available or worth buying; best-of-N reranking with a PRM
(Snell's mechanism 1, §6.1).

**Cost.** Data: step labels are the most expensive supervision here
(PRM800K cost OpenAI-scale annotation). Compute: PRM training +
best-of-N inference.

**HSEA fit: ADAPT (verifier + reranker, not trainer).** Train dispatch
*policies* with RLVR outcome rewards (§3.1); train a lightweight PRM
on L2 step events (action/edge/score) as a **test-time verifier** for
best-of-N dispatch/vignette selection and as an abstention signal
(low step-confidence → escalate).

**Experiment E-PRM.** Label L2 trajectory steps auto (§5.2) → train
PRM head → best-of-8 dispatch rerank vs greedy on holdout; report
BoN gain + calibration (ECE) of step scores as an escalate trigger.

### 5.2 Math-Shepherd (automatic process supervision)

**What.** A process reward model trained on **automatically
constructed** step supervision (no human step labels): per-step reward
scores used for reranking and step-by-step PPO. Mistral-7B: GSM8K
77.9→84.1% (→89.1% with verification), MATH 28.6→33.0% (→43.5%)
[Wang et al.](#c-shepherd).

**When it wins.** Human step labels are unaffordable but outcome
labels + rollouts exist (completions-per-step Monte-Carlo
estimation).

**Cost.** Data: outcome-labeled prompts only. Compute: sampling-heavy
label construction, then standard PRM/PPO.

**HSEA fit: ADOPT the labeling trick for E-PRM.** HSEA L2 logs have
outcome labels (score events) but no step labels; Math-Shepherd-style
step-credit estimation turns 250+ RLVR/L2 episodes into step training
data for free.

**Experiment.** Step 1 of E-PRM: implement per-step success-rate
estimation over L2 trajectories; validate step labels against a
50-episode human spot-check before PRM training.

## 6. Test-time training and scaling

### 6.1 Compute-optimal test-time scaling (Snell et al.)

**What.** Two mechanisms to scale inference compute: (1) search
against dense process-based verifier RMs (best-of-N / beam search);
(2) adaptively updating the response distribution at test time. Key
finding: effectiveness varies by prompt difficulty, so a
**compute-optimal** per-prompt allocation beats fixed budgets by 4×
over best-of-N; FLOPs-matched, test-time compute lets a small model
outperform a 14× larger one where the small model has non-trivial base
success [Snell et al.](#c-snell).

**When it wins.** Hard prompts are rare: spend verifier/search budget
only where difficulty is high; serving-time quality without training.

**Cost.** Data: none (uses existing verifiers). Compute: inference
multiplier N (best-of-N) or search budget — pay at serving, not
training.

**HSEA fit: ADOPT at eval/serving time.** Dispatch decisions and
abstention calls are exactly the rare-hard-prompt regime: greedy
decode by default, best-of-N + PRM/verifier rerank (§5.1) when stakes
are high or confidence low. Zero training cost, immediate Gate 3
leverage (escalation recall via verifier disagreement).

**Experiment E-TTC.** On dispatch holdout + abstention eval:
greedy vs best-of-8-with-verifier vs compute-optimal (verifier only if
top-1 confidence < τ); report accuracy, abstain/escalate recall, and
mean N (cost). See
[fig-testtime-verify.png](alignment-methods-posttraining/fig-testtime-verify.png).

### 6.2 s1: Simple test-time scaling (budget forcing)

**What.** Minimal recipe: SFT Qwen2.5-32B-Instruct on **s1K** (1,000
reasoning traces curated for difficulty/diversity/quality) + **budget
forcing** — terminate the thinking process early or extend it by
appending "Wait" so the model double-checks. s1-32B exceeds o1-preview
on MATH/AIME24 by up to 27%; scaling test-time compute extrapolates
AIME24 50→57% [Muennighoff et al.](#c-s1).

**When it wins.** Small curated reasoning sets + controllable thinking
budgets; sequential scaling complements parallel (best-of-N) scaling.

**Cost.** Data: ~1k traces. Compute: SFT-cheap + inference tokens for
extended thinking.

**HSEA fit: ADAPT for high-stakes dispatch reasoning.** Add a
verifier-triggered "Wait"-style second pass on dispatch/vignette
prompts (re-check priority/citations before answering); curate a 1k
HSEA reasoning-trace set in the s1K style (difficulty/diversity/
quality) from verifier-passing rollouts.

**Experiment E-BUDGET.** s1K-style 1k-trace SFT on the 27B lane +
budget-forcing at inference vs E-TTC best-of-N on dispatch holdout;
compare accuracy-per-inference-FLOP and escalation recall.

## 7. Preference-data curation

### 7.1 UltraFeedback (scale + diversity + debiasing)

**What.** 1M+ GPT-4 feedback annotations over 250k diverse
user-assistant conversations; identifies **scale and diversity** as
the key factors and applies bias-mitigation (length/position) for
reliable AI feedback. LLaMA aligned on it excels at chat benchmarks
[Cui et al.](#c-ultrafeedback). Also the training set behind ORPO's
small-model wins (§2.3).

**When it wins.** Bootstrapping a preference corpus from scratch with
AI annotators; diversity across skills matters more than annotator
pedigree.

**Cost.** Data: prompt diversity + AI annotation budget (API $, no
human labeling). Compute: standard DPO/SimPO.

**HSEA fit: ADAPT the template.** HSEA's 7,000 dispatch pairs already
mirror the (situation, chosen, rejected) shape; apply the UltraFeedback
playbook to grow them: diversify situations (rule-shifted variants,
multi-report pileups), multi-aspect AI grading (correctness, doctrine
citation, terseness), debiased labels.

**Experiment E-PREFSCALE.** Generate 3,000 diversified dispatch
situations (paraphrase + threshold-shift + multi-report) with AI
pairwise labels; DPO/SimPO ablation: base 7k vs 7k+3k on dispatch
holdout + OOD shifts.

### 7.2 RLAIF (AI-labeled preferences)

**What.** Train the reward model on preferences from an off-the-shelf
LLM instead of humans: RLAIF matches RLHF on summarization/helpful
dialogue and **beats** RLHF on harmless dialogue; self-improvement
works even when the labeler is the same checkpoint as the policy; and
direct-RLAIF (LLM scores during RL, no RM) beats canonical RLAIF [Lee
et al.](#c-rlaif).

**When it wins.** Human labels are the bottleneck; harmlessness
judgments (easier to specify than helpfulness) favor AI labelers.

**Cost.** Data: prompts + AI labeler inference. Compute: RM/RL
standard; d-RLAIF skips RM training.

**HSEA fit: ADOPT for constitution-grounded label scale-up.** The
playbook's RLAIF step gets its evidence here: a doctrine-prompted
frontier labeler grades dispatch/abstention candidate pairs, with
human audit on a sampled subset. Prefer direct scoring (d-RLAIF
style) over training a separate RM (R1's anti-neural-RM lesson).

**Experiment E-RLAIF.** AI-label 2,000 dispatch pairs with a
doctrine-grounded rubric; measure labeler-human agreement on 200
audited pairs; SimPO on AI vs human labels; adopt AI labels if
agreement ≥ 0.85 and downstream Δ within noise.

### 7.3 Constitutional AI (self-critique + revision + RLAIF)

**What.** Two phases from a written constitution (principles list) and
no human harm labels: supervised self-critique-and-revision finetune,
then RL from AI feedback against an AI preference model — yielding a
harmless-but-non-evasive assistant that explains objections [Bai et
al.](#c-cai). This is the recipe the playbook's mini-constitution step
implements.

**When it wins.** Harm/safety behavior must be specified in prose
(doctrine) rather than examples; evasiveness must stay low (explain,
don't stonewall — the HSEA escalate action).

**Cost.** Data: constitution + red-team prompts + AI critiques.
Compute: SFT + RLAIF standard.

**HSEA fit: ADOPT (already planned; keep).** No change: encode
disaster-ops doctrine (ICS/FEMA/US&R + HSEA abstain/escalate policy)
as the constitution, self-critique dispatch/abstention drafts against
it, RLAIF per §7.2.

**Experiment E-CAI.** A/B the mini-constitution: with vs without an
explicit "escalate-with-reasons" principle; grade escalation recall +
evasiveness (benign compliance + explanation presence) on the 60
abstention scenarios.

## 8. Multi-turn and agent RL

### 8.1 Search-R1 (RL-only tool learning + retrieved-token masking)

**What.** The LLM learns **solely through RL** to issue multiple
search queries mid-reasoning with real-time retrieval — no supervised
tool-use data. Two load-bearing tricks: retrieved-token masking (no
loss on retrieved tokens → stable training) and a simple
outcome-based reward. +26% (Qwen2.5-7B), +21% (3B), +10% (LLaMA3.2-3B)
over SOTA on seven QA datasets [Jin et al.](#c-searchr1).

**When it wins.** Tools/observations interleave with reasoning;
supervised tool cloning is unavailable or insufficient (the v1 null
lesson, §8.4).

**Cost.** Data: task prompts + outcome checker. Compute: multi-turn
rollout RL (GRPO-class); masking is free.

**HSEA fit: ADOPT the two tricks for AgentTune.** HSEA vignette/L2
rollouts interleave model actions with environment observations
exactly like Search-R1's search results: mask observation tokens in
the loss, reward verifiable outcomes (dispatch correct, victim found,
no safety violation). This is the how-to for the AgentTune stage.

**Experiment.** Fold into E-RLVR phase 2: multi-turn vignette
rollouts with observation masking vs without (ablation); report
reward variance + task success.

### 8.2 SWE-RL (rule-reward RL on software evolution)

**What.** First scaled RL-reasoning approach for real software
engineering: lightweight **rule-based reward** (similarity between
ground-truth and generated solutions) lets the LLM autonomously
recover developer reasoning from open-source evolution data (issues,
snapshots, changes). Llama3-SWE-RL-70B: 41.0% on SWE-bench Verified
[Wei et al.](#c-swerl).

**When it wins.** Rich trajectory/provenance logs exist with
ground-truth endpoints; reasoning must be recovered, not just cloned.

**Cost.** Data: evolution logs (large but free). Compute: scaled RL.

**HSEA fit: ADAPT the reward idea.** HSEA's analogue of "ground-truth
solution similarity" is graded-execution similarity: reward vignette
rollouts by match to gold dispatch/plan fields (priority, unit,
doctrine citation) rather than by a learned RM. Directly applicable
to the 250 RLVR rows + future L2 corpora.

**Experiment E-SWEHSEA.** On L2 vignette logs: trajectory reward =
weighted field-match to gold plan; GRPO vs SFT-clone on held-out
vignettes; report success + OOD rule-shift (Chu-style).

### 8.3 ToRL (tool-integrated RL from base models)

**What.** RL **directly from base models** (no post-training)
discovers tool-use strategies via unrestricted exploration: ToRL-7B
hits 43.3% on AIME24, +14% over RL without tools and +17% over the
best supervised tool-integrated-reasoning model, with emergent
strategic tool invocation and self-regulation [Li et
al.](#c-torl). (Paper under CC BY 4.0 — see FIGURES.md.)

**When it wins.** Tool use must be discovered (when/how to call),
not imitated; supervised TIR plateaus.

**Cost.** Data: verifiable tasks + tool sandbox. Compute: scaled RL
from base (expensive at full scale; the finding transfers).

**HSEA fit: ADAPT the finding, not the from-base recipe.** HSEA
starts from instruct models, not base — but ToRL is the strongest
evidence that tool-calling dispatch/vignette behavior should be
**RL-trained with verifiable rewards**, not SFT-cloned: it explains
the flywheel-v1 null (single-teacher cloning) and justifies E-RLVR
phase 2 over more cloning.

**Experiment.** Evidence input to E-RLVR/E-SWEHSEA design (no
standalone run): require every AgentTune proposal to state its
verifier + exploration budget, not just its demo set.

### 8.4 SFT memorizes, RL generalizes (Chu et al.)

**What.** Controlled comparison on rule-based textual + visual
variants: outcome-reward RL generalizes across rule shifts and visual
OOD (and improves underlying visual recognition), while SFT memorizes
training data and collapses OOD. Counter-balance: SFT still helps by
stabilizing output format so subsequent RL can gain [Chu et
al.](#c-chu).

**When it wins (as a principle).** Anytime OOD robustness is graded:
keep SFT for format, RL for behavior.

**Cost.** A design constraint, not a method — free.

**HSEA fit: ADOPT as the loop's ordering principle.** It explains v1
(success-filtered cloning → null held-out delta) and prescribes v2:
SFT for the typed dispatch/abstain contract → RLVR/GRPO for the
behavior → OOD rule-shift evals (shifted triage thresholds, unseen
sites) as the generalization gate.

**Experiment.** Every training experiment in this note (E-*) includes
an OOD rule-shift arm by default; see §12 for the formal shift
metrics.

### 8.5 Cross-cutting experiment: preference-stage bake-off (E-DPO)

Four arms on identical data (7,000 dispatch pairs + refusal/abstention
pairs, Qwen3.8-27B-LoRA, same seed/schedule): (a) length-normalized
DPO, (b) SimPO, (c) KTO on binarized outcomes, (d) IPO if available.
Primary: dispatch priority accuracy (holdout + OOD shift) + Gate 1
(no-regression refusal) + benign compliance + abstention recall.
Secondary: training memory/time, response length. Winner becomes the
AlignTune default; all four checkpoints feed E-RLVR ablations. Full
protocol in follow-up issue #2. Decision map:
[fig-preference-map.png](alignment-methods-posttraining/fig-preference-map.png).

## 9. Refusal and safety training that preserves utility

### 9.1 Safety-tuning mix (Bianchi et al.)

**What.** Adding just ~3% safety examples (a few hundred demos) to
instruction tuning substantially improves safety without degrading
capability/helpfulness on standard benchmarks — but too much
safety-tuning causes **exaggerated safety** (refusing safe prompts
that superficially resemble unsafe ones) [Bianchi et
al.](#c-bianchi).

**When it wins.** Every finetune needs the safety mix; the dose is the
whole game (hundreds, not thousands).

**Cost.** Data: a few hundred safety demos. Compute: negligible delta.

**HSEA fit: ADOPT (already planned; dose it).** Keep the "100s of
safety examples in every finetune" rule; pair every safety-mix change
with an over-refusal read (§9.3) so SafeTune repairs can't hide
behind refusal-everything.

**Experiment E-SAFEMIX.** SFT with {0, 100, 300, 1000} safety demos:
plot refusal rate (HarmBench/AIR-Bench subsets) vs over-refusal rate
(XSTest-style + abstention comply items) vs dispatch accuracy; pick
the knee for the standard mix.

### 9.2 Fine-tuning drift warning (Qi et al.)

**What.** Safety alignment breaks under finetuning: 10 adversarial
examples jailbreak GPT-3.5 Turbo for <$0.20; worse, even **benign**
common datasets (Alpaca, Dolly, LLaVA-Instruct) measurably degrade
safety alignment [Qi et al.](#c-qi).

**When it wins (as a constraint).** Every finetune, including benign
ones, must re-prove refusal — the empirical basis for HSEA's
re-run-the-stack-after-every-stage rule.

**Cost.** Eval cost only (the safety stack runs).

**HSEA fit: ADOPT as Gate 1's rationale (already planned; keep).**
No change: Gate 1's no-regression clause (within 0.02 of base) IS this
paper operationalized. Cite it in every finetune PR's risk section.

**Experiment.** None new: Gate 1 + the drift audit before/after every
E-* run.

### 9.3 XSTest (over-refusal measurement)

**What.** 250 safe prompts across ten types (that well-calibrated
models must NOT refuse) + 200 unsafe contrasts (that they should):
systematically exposes exaggerated-safety failures and lexical
overfitting [Röttger et al.](#c-xstest). (Paper under CC BY 4.0 —
see FIGURES.md.)

**When it wins.** Any safety-tuning change needs its counter-metric;
safe/unsafe contrast pairs separate calibration from
refusal-everything.

**Cost.** Eval-only (450 prompts).

**HSEA fit: ADOPT the contrast-pair pattern.** Mirror it in-domain:
for every abstention/refusal scenario, curate a benign lookalike
(same disaster vocabulary, safe request) — the abstention `comply`
split already starts this; grow it to full contrast coverage.

**Experiment E-XS.** Extend the 60 abstention scenarios with
contrastive comply lookalikes (target: 1:1); report the safe/unsafe
refusal pair (§12) for every E-* checkpoint.

### 9.4 WildJailbreak (adversarial safety data at scale)

**What.** WildTeaming mines in-the-wild jailbreak tactics into 5.7K
tactic clusters and composes them into attacks (4.6× more diverse and
successful than prior jailbreak methods); WildJailbreak packages
262K vanilla + adversarial prompt-response pairs, including benign
lookalikes that resemble harmful queries — explicitly to mitigate
exaggerated safety during safety training [Jiang et
al.](#c-wild).

**When it wins.** Safety training needs adversarial coverage + benign
contrasts at a scale humans can't write; open weights need open safety
data (frontier safety data stays closed).

**Cost.** Data: mining + synthesis pipeline (API $; one-time).
Compute: standard SFT-mix training.

**HSEA fit: ADAPT the pipeline to disaster misuse.** HSEA's redteam
set (`data/redteam/`) is the right target: mine disaster-specific
misuse tactics (fake evacuation orders, rescue fraud, surveillance
abuse), synthesize vanilla + adversarial variants with benign
lookalikes, and mix into SafeTune repair. Do NOT import WildJailbreak
wholesale (generic chat distribution; license-check any subset first).

**Experiment E-WILD.** Build 500 disaster-misuse adversarial variants
+ 500 benign lookalikes in the WildJailbreak 4-type schema; SafeTune
repair ablation (with/without) on redteam ASR vs benign compliance.

## 10. Comparison table

Verdicts: ADOPT = do it; ADAPT = do a scoped variant; SKIP = don't
(default PPO only). Cost: $ = eval/inference-only, $$ = SFT-class,
$$$ = online-RL-class.

| Method | Slot | Needs | Verdict | Why in one line |
|---|---|---|---|---|
| Data-centric SFT (§1.1) | AlignTune | 1k demos | ADOPT | Format/contract teaching; curation beats scale |
| RFT / verifier-filtered SFT (§1.2) | Baseline | Verifier + N× infer | ADOPT | Mandatory baseline every RLVR pilot must beat |
| Length-norm DPO (§2.1) | AlignTune pref | Pairs | ADOPT | Tülu-3-vetted default; add on-policy pairs |
| IPO (§2.2) | Bake-off arm | Pairs | ADAPT | Insurance vs DPO overfit on templated pairs |
| ORPO (§2.3) | Edge lanes | Pairs | ADAPT | Single-stage recipe for Tiny Aya / E2B |
| SimPO (§2.4) | AlignTune pref | Pairs | ADOPT | Ref-free, length-controlled; fits 1×80GB |
| KTO (§2.5) | Unpaired slot | Binary labels | ADOPT | L2/abstention labels are binary, not paired |
| RLVR safety rewards (§3.1) | AgentTune | Verifier | ADOPT | Top pick: trains checkable safety behavior |
| R1 pattern (§3.2) | AgentTune/edge | Traces | ADAPT | Rule rewards + cold start + distill to edge |
| PPO (§4.1) | Reference only | RM + critic | SKIP | Dominated by GRPO/ReMax for verifier tasks |
| GRPO (§4.2) | AgentTune RL | Verifier | ADOPT | Critic-free online RL; AgentTune-native |
| ReMax (§4.3) | Fallback RL | Reward | ADAPT | Single-sample RL if GRPO unstable on traces |
| PRM rerank (§5.1) | Test-time | Step labels | ADAPT | Verifier/reranker + escalate signal, not trainer |
| Math-Shepherd (§5.2) | Labeling | Outcome logs | ADOPT | Free step labels from L2 outcome logs |
| Test-time scaling (§6.1) | Serving/eval | Verifier | ADOPT | Best-of-N + verifier; zero training cost |
| Budget forcing (§6.2) | Serving/eval | 1k traces | ADAPT | Second-pass recheck on high-stakes prompts |
| UltraFeedback template (§7.1) | CuratorKIT | API $ | ADAPT | Diversify + debias the 7k dispatch pairs |
| RLAIF (§7.2) | CuratorKIT | Labeler | ADOPT | Doctrine-grounded label scale-up; direct scores |
| CAI (§7.3) | CuratorKIT | Constitution | ADOPT | Mini-constitution recipe (already planned) |
| Search-R1 tricks (§8.1) | AgentTune | — | ADOPT | Mask observation tokens; outcome rewards |
| SWE-RL rewards (§8.2) | AgentTune | Gold plans | ADAPT | Field-match trajectory rewards for vignettes |
| ToRL finding (§8.3) | Design rule | — | ADAPT | RL>clone for tool use; require verifier + budget |
| SFT-mem/RL-gen (§8.4) | Ordering | — | ADOPT | SFT format → RL behavior → OOD-shift gates |
| Safety mix (§9.1) | Every stage | 100s demos | ADOPT | Dose at knee of refusal/over-refusal curve |
| Drift audits (§9.2) | Gates | Eval runs | ADOPT | Gate 1 rationale (already planned) |
| XSTest contrasts (§9.3) | Eval | 450 prompts | ADOPT | Over-refusal counter-metric for every change |
| WildJailbreak pipe (§9.4) | SafeTune | API $ | ADAPT | Disaster-misuse adversarial + lookalike mix |

## 11. HSEA recommendations

Ordered by leverage per dollar (all $ figures are Modal/API spend;
human curation time dominates and is noted where it binds):

1. **E-RLVR pilot first (GRPO + verifiable safety-procedure rewards,
   §3.1 + §4.2 + §8.1).** Reuses existing data and scorers; trains
   the exact behaviors the gates grade; falsifiable against E-RFT.
   Follow-up issue #1.
2. **E-DPO preference bake-off (length-norm DPO vs SimPO vs KTO,
   §8.5).** One training config × four losses on the same 7k pairs;
   settles the AlignTune default for all lanes. Follow-up issue #2.
3. **Test-time verifier stack now (best-of-N + PRM rerank + XSTest
   contrasts, §5–§6 + §9.3).** Zero training cost; improves Gate 3
   escalation recall and every future checkpoint comparison.
4. **Label flywheel (RLAIF + CAI + UltraFeedback template, §7).**
   Scales dispatch/abstention preference data 7k → 10k+ with audited
   AI labels; unblocks (1)–(2) from data ceiling.
5. **Safety-dose calibration (E-SAFEMIX + E-WILD, §9.1 + §9.4).**
   Finds the refusal/over-refusal knee and hardens the redteam set
   before any release claim.
6. **Edge-lane recipe (ORPO vs R1-distill, §2.3 + §3.2).** Settles
   Tiny Aya / Gemma E2B training after the 27B pilots land; do not
   run first.

Explicit non-goals: default PPO (dominated), wholesale WildJailbreak
import (distribution + license), neural reward models where a
deterministic checker exists (R1's hacking lesson), more SFT cloning
scale as a substitute for RL (v1 null + Chu).

## 12. Metrics: native inventory + HSEA derivations

### 12.1 Native metrics per method (as reported in the papers)

| Method | Native metrics |
|---|---|
| SFT / LIMA | Human preference vs GPT-4/Bard/DaVinci003; format adherence |
| RFT | Verifier pass@N → filtered-SFT accuracy |
| DPO | Win rate vs SFT/PPO; reward margin; sentiment/summary quality |
| IPO | Illustrative-case win vs DPO; theory guarantees |
| ORPO | AlpacaEval 2.0, IFEval (loose), MT-Bench |
| SimPO | AlpacaEval 2 LC-WR, Arena-Hard WR, MT-Bench, length |
| KTO | Win rates (1B–30B) on binary-labeled tasks |
| RLVR | GSM8K / MATH / IFEval deltas |
| R1 | AIME/MATH/Code pass@1; aha-behavior counts; distill parity |
| PPO | Episode reward, KL(π‖ref), value loss |
| GRPO | Task accuracy (MATH/GSM8K), GPU memory, group stats |
| ReMax | AlpacaEval WR, MT-Bench, GPU memory (−46%), wall-time |
| PRM | MATH solve rate, best-of-N gain, step-label accuracy |
| Math-Shepherd | GSM8K/MATH (+PPO, +verify) |
| Test-time scaling | FLOPs-matched accuracy, 4× efficiency over BoN |
| s1 | AIME24/MATH, scaling slope (50→57%), sample count |
| UltraFeedback | Chat-bench scores of aligned model |
| RLAIF | Human preference % vs RLHF/SFT; agreement rate |
| CAI | Human harmlessness/helpfulness; evasiveness |
| Search-R1 | 7-QA-dataset accuracy (+26%/+21%/+10%) |
| SWE-RL | SWE-bench Verified solve rate (41.0%) |
| ToRL | AIME24 (43.3%), code ratio, pass ratio |
| SFT-vs-RL | OOD rule-shift accuracy (text + visual) |
| Safety mix | Safety rate + capability retention + XSTest |
| Drift | Post-FT harmfulness/ASR vs pre-FT |
| XSTest | Refusal rate on 250 safe vs 200 unsafe |
| WildJailbreak | ASR (vanilla/adversarial), over-refusal rate |

### 12.2 HSEA derivations (formal definitions)

Existing HSEA metrics keep their definitions
([metrics](metrics.md), [safety-gates](safety-gates.md)). New
derivations proposed here (all deterministic, all computable from
Inspect-ai logs + manifests):

- **dispatch_priority_acc** = correct-P1/P2/P3 fraction over dispatch
  situations. Native to none of the papers (domain metric); primary
  outcome for E-DPO/E-RLVR.
- **abstain_recall** = P(output=abstain | label=abstain);
  **escalate_recall** = P(output=escalate | label=escalate) (Gate 3
  already). Split the current `abstention_rate` by label.
- **over_refusal_rate** = P(output∈{refuse} | label=comply) over
  abstention comply items + XSTest-style lookalikes (E-XS). XSTest
  derivation; report paired with refusal rate always.
- **confirm_rate** = P(confirmation request precedes an irreversible
  tool call | rollout plans one). RLVR reward (c); parse from
  transcripts with the typed tool schema.
- **verifier_pass_rate** = fraction of rollouts passing ALL
  deterministic checks (priority match + label match + confirm +
  format). RLVR outcome metric; the RFT-vs-RL comparison number.
- **bon_gain** = acc(best-of-N by verifier) − acc(greedy), N=8.
  Snell/PRM derivation; E-TTC primary with mean-N cost attached.
- **ood_shift_gap** = acc(holdout) − acc(rule-shifted holdout), where
  the shift re-thresholds triage cutoffs or swaps site/entity names
  (Chu derivation). Every E-* reports it; RL should shrink it vs SFT.
- **labeler_agreement** = Cohen's κ between AI and human labels on
  the 200-pair audit (E-RLAIF gate κ ≥ 0.85 … reported as raw
  agreement in the issue for small-n honesty).
- **step_score_ece** = ECE of PRM step scores vs step success (E-PRM);
  calibrates the escalate trigger threshold τ.
- **safety_dose_curve** = (refusal_rate, over_refusal_rate,
  dispatch_priority_acc) at safety-mix ∈ {0, 100, 300, 1000}
  (E-SAFEMIX); the knee sets the standard mix.

## 13. Citations

All arXiv links below were verified against primary paper content
(HTML/PDF). Venues shown only where confirmed; otherwise the arXiv
year stands as the date.

- <a id="c-ppo"></a>Schulman, Wolski, Dhariwal, Radford, Klimov.
  Proximal Policy Optimization Algorithms. 2017.
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- <a id="c-dpo"></a>Rafailov, Sharma, Mitchell, Ermon, Manning, Finn.
  Direct Preference Optimization: Your Language Model is Secretly a
  Reward Model. NeurIPS 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- <a id="c-ipo"></a>Azar, Guo, Hariharan, Czarnecki, Rowland, et al.
  A General Theoretical Paradigm to Understand Learning from Human
  Preferences. ICML 2024.
  [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- <a id="c-cai"></a>Bai, Kadavath, Sanderson, Askell, Herbert-Voss, et
  al. Constitutional AI: Harmlessness from AI Feedback. 2022.
  [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- <a id="c-lima"></a>Zhou, Liu, Xu, Iyer, Sun, et al. LIMA: Less Is
  More for Alignment. NeurIPS 2023.
  [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- <a id="c-prm"></a>Lightman, Kosaraju, Burda, Edwards, Baker, et al.
  Let's Verify Step by Step. ICLR 2024.
  [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- <a id="c-xstest"></a>Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy.
  XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours
  in Large Language Models. NAACL 2024. CC BY 4.0.
  [arXiv:2308.01263](https://arxiv.org/abs/2308.01263)
- <a id="c-bianchi"></a>Bianchi, Suzgun, Attanasio, Röttger, Jurafsky,
  Hashimoto, Zou. Safety-Tuned LLaMAs: Lessons From Improving the
  Safety of Large Language Models that Follow Instructions. ICLR 2024.
  [arXiv:2309.07875](https://arxiv.org/abs/2309.07875)
- <a id="c-rlaif"></a>Lee, Phatale, Mansoor, Mesnard, Ferret, et al.
  RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback
  with AI Feedback. ICML 2024.
  [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
- <a id="c-qi"></a>Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson.
  Fine-tuning Aligned Language Models Compromises Safety, Even When
  Users Do Not Intend To! 2023.
  [arXiv:2310.03693](https://arxiv.org/abs/2310.03693)
- <a id="c-remax"></a>Li, Li, Liu, Chen, Li, et al. ReMax: A Simple,
  Effective, and Efficient Reinforcement Learning Method for Aligning
  Large Language Models. ICML 2024.
  [arXiv:2310.10505](https://arxiv.org/abs/2310.10505)
- <a id="c-ultrafeedback"></a>Cui, Yuan, Ding, Yao, He, et al.
  UltraFeedback: Boosting Language Models with Scaled AI Feedback.
  ICML 2024.
  [arXiv:2310.01377](https://arxiv.org/abs/2310.01377)
- <a id="c-shepherd"></a>Wang, Deng, Chen, Ai, et al. Math-Shepherd:
  Verify and Reinforce LLMs Step-by-step without Human Annotations.
  2023.
  [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)
- <a id="c-orpo"></a>Hong, Lee, Thorne. ORPO: Monolithic Preference
  Optimization without Reference Model. EMNLP 2024.
  [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
- <a id="c-kto"></a>Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela. KTO:
  Model Alignment as Prospect Theoretic Optimization. ICML 2024.
  [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- <a id="c-deepseekmath"></a>Shao, Wang, Zhu, Xu, Song, et al.
  DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
  Language Models. ICML 2024.
  [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- <a id="c-simpo"></a>Meng, Xia, Chen. SimPO: Simple Preference
  Optimization with a Reference-Free Reward. 2024.
  [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- <a id="c-wild"></a>Jiang, Rao, Han, Ettinger, Brahman, et al.
  WildTeaming at Scale: From In-the-Wild Jailbreaks to
  (Adversarially) Safer Language Models. NeurIPS 2024.
  [arXiv:2406.18510](https://arxiv.org/abs/2406.18510)
- <a id="c-snell"></a>Snell, Lee, Xu, Kumar. Scaling LLM Test-Time
  Compute Optimally can be More Effective than Scaling Model
  Parameters. 2024.
  [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- <a id="c-agentharm"></a>Andriushchenko, Souly, Dziemian, Duenas, Lin,
  et al. AgentHarm: A Benchmark for Measuring Harmfulness of LLM
  Agents. 2024.
  [arXiv:2410.09024](https://arxiv.org/abs/2410.09024)
- <a id="c-tulu3"></a>Lambert, Morrison, Pyatkin, Huang, Ivison, et al.
  Tülu 3: Pushing Frontiers in Open Language Model Post-Training.
  2024.
  [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
- <a id="c-r1"></a>DeepSeek-AI (Guo, Yang, Zhang, et al.).
  DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning. Nature 645:633–638 (2025); preprint 2025.
  [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- <a id="c-s1"></a>Muennighoff, Yang, Shi, Li, Fei-Fei, et al. s1:
  Simple test-time scaling. 2025.
  [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)
- <a id="c-chu"></a>Chu, Zhai, Yang, Tong, Xie, et al. SFT Memorizes,
  RL Generalizes: A Comparative Study of Foundation Model
  Post-training. ICML 2025.
  [arXiv:2501.17161](https://arxiv.org/abs/2501.17161)
- <a id="c-swerl"></a>Wei, Duchenne, Copet, Carbonneaux, Zhang, et al.
  SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open
  Software Evolution. 2025.
  [arXiv:2502.18449](https://arxiv.org/abs/2502.18449)
- <a id="c-searchr1"></a>Jin, Zeng, Yue, Yoon, Arik, et al. Search-R1:
  Training LLMs to Reason and Leverage Search Engines with
  Reinforcement Learning. 2025.
  [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- <a id="c-torl"></a>Li, Zou, Liu. ToRL: Scaling Tool-Integrated RL.
  2025. CC BY 4.0.
  [arXiv:2503.23383](https://arxiv.org/abs/2503.23383)
