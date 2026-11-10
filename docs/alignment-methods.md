# HSEA alignment methods: unified index

One front door to the three alignment survey lanes. Each lane is a
standalone survey note with its own branch, figures, report, and
follow-up issues; this index summarizes each lane in one paragraph,
ranks the combined recommendations with effort/impact/cost, and gives
the reading order for future agents. Audit trail (coverage map, dedup
findings, citation and number checks):
[reports/alignment-synthesis](reports/alignment-synthesis.md).

Lane-note merge status: each lane doc below lands via its own branch
and is read on that branch until merged (`git show
origin/<branch>:docs/<note>`). This index merges after the lanes.

| Lane | Note | Branch | Issues |
|---|---|---|---|
| Post-training | [alignment-methods-posttraining](alignment-methods-posttraining.md) | `followup/alignment-posttraining` | #48, #49 |
| VLM / footage | [alignment-methods-vlm](alignment-methods-vlm.md) | `followup/alignment-vlm` | #50, #51 |
| Safety | [alignment-methods-safety](alignment-methods-safety.md) + [safety-metrics](safety-metrics.md) | `followup/alignment-safety` | #46, #47 |

Context this index assumes: [research-track](research-track.md) (lanes
+ loop), [process](process.md) (commands),
[alignment-research](alignment-research.md) §4 (playbook),
[safety-gates](safety-gates.md) ( Gate 1/2/3 bars).

## Lane summaries

**Post-training (text/agent alignment).** Surveys 27 methods across 9
families — SFT variants, the DPO family (DPO, IPO, ORPO, SimPO, KTO),
RLVR with verifiable rewards, online RL (PPO skipped, GRPO adopted,
ReMax fallback), process supervision/PRMs, test-time scaling,
preference-data curation (UltraFeedback, RLAIF, CAI), multi-turn/agent
RL, and refusal/safety-preserving utility — each with what/when/cost,
an adopt/adapt/skip verdict, and a concrete experiment on
`data/flywheel-v2/` (SFT 8,673 / DPO 7,000 / RLVR 250, per
[issue-7 curation](reports/issue-7-flywheel-v2-curation.md)) or
`data/abstention/` (60 risk scenarios + 20 benign, per
[issue-21](reports/issue-21-abstention.md)). Top pick: the E-RLVR pilot
(GRPO + deterministic safety-procedure rewards, no neural RM) filed as
#48, with the E-DPO preference bake-off as #49.

**VLM / footage alignment.** Surveys 21 vision-language methods across
7 families — instruction-tuning mixes, region/grounding objectives,
multimodal DPO/RLHF, vision RLVR, video-temporal, hallucination
reduction, small-VLM/edge — every verdict mapped to #24-cleared
corpora and the `footage_score` scorer. Lead with the eval axis first
(DisasterM3 referring task, box-IoU/pointing metrics, HSEA-POPE,
HaloQuest abstain probes, filed as #50), then the vision-RLVR pilot on
FloodNet counting + LADI severity (#51), then on-policy footage DPO;
video training stays blocked (no cleared corpus) and edge work waits
for the workhorse loop.

**Safety alignment + metrics.** Surveys 10 safety methods
(deliberative alignment, constitutional classifiers, over-refusal
measurement, calibration, abstention/deferral, graded jailbreak eval +
circuit breakers, agent-misuse benches, injection defenses, trace
faithfulness, safety-vs-capability Pareto tracking) with HSEA-fit
verdicts, and pairs the survey with
[safety-metrics](safety-metrics.md): a full inventory of existing
safety metrics with baselines traced to #10/#14/#21/#22, plus six
derived high-stakes metrics (M1 EHWR, M2 escalation F2, M3 worst-slice
robustness, M4 recovery rate, M5 severity-weighted ECE, M6 refusal
consistency) with formal definitions. Ships as #46 (graded ASR + M6)
and #47 (M1/M2/M3/M5 for abstention).

## Cross-lane ranking

Ranked by leverage per dollar. Effort: S/M/L in code + curation +
GPU. Cost: $0 (no GPU), API $ (inference/labeling only), $ (Modal
LoRA pilots, pre-approved against the $50 cap). Details per item live
in the lane notes; the synthesis report records why this order.

| Rank | Action | Effort | Impact | Cost |
|---|---|---|---|---|
| 1 | Eval tranche first: #50 grounding/POPE/probes + #46 graded-ASR/R-CONS + #47 EHWR/SW-ECE/WSR/F2 | M (scorer code + curation) | Unlocks every training claim; fixes binary-ASR overstatement, flat-risk degeneracy, missing grounding axis | $0 GPU (API evals) |
| 2 | RLVR pilots: #48 text safety-procedure + #51 vision counting/severity, one shared GRPO recipe | L (verifiers + Modal runs) | Trains checkable safety/vision behavior; OOD generalization; fixes verbose-counting failure | $ (LoRA pilots) |
| 3 | Preference default: #49 bake-off (length-norm DPO vs SimPO vs KTO) | M (4 arms, one config) | Settles the AlignTune default for all lanes; feeds #48 | $ (LoRA x4) |
| 4 | $0 wins now: Pareto tracking, best-of-N + verifier, XSTest contrasts, on-policy discipline | S | Kills safetywashing misreads; Gate-3 recall lift with zero training | $0 |
| 5 | Label flywheel: RLAIF + CAI + UltraFeedback template, 7k to 10k+ pairs | M (API + audit) | Unblocks ranks 2-3 from the data ceiling | API $ |
| 6 | Safety-dose calibration: E-SAFEMIX knee + E-WILD disaster-misuse mix | M | Refusal/over-refusal operating point; hardened redteam set pre-release | $ + API $ |
| 7 | Deliberative rubric pilot on the small lane | S-M | Highest-leverage abstention training bet; rubric is the spec | API $ + CPU |
| 8 | Data growth: hard-benign, risk-ID probes, injection-utility pairs, disciplined footage SFT | M (curation) | Resolves the live over-refusal gap; completes the AgentDojo triple | $0 to API $ |
| 9 | Edge lane last: ORPO vs R1-distill, SmolVLM SFT, guarded FastV | M | Edge recipe, only after 27B pilots land | $ (small) |

Sequencing rules: rank 1 before ranks 2-3 (no training claim without
meters); rank 4 anytime, starting now; rank 5 feeds 2-3; rank 6 before
any release claim; rank 9 last. Explicit non-goals carried over from
the lanes: default PPO, wholesale WildJailbreak import, neural reward
models where a deterministic checker exists, more SFT cloning as an RL
substitute, video training before a cleared corpus lands.

## Reading order for future agents

1. [research-track](research-track.md) + [process](process.md) +
   [alignment-research](alignment-research.md) §4 — mission, loop,
   playbook, commands.
2. This index — what exists and what to do first.
3. [metrics](metrics.md) unified map + [safety-metrics](safety-metrics.md)
   — the meters, implemented vs proposed, with baselines.
4. Your lane note (post-training for text/agent training, VLM for
   footage, safety for eval/hardening) — methods, verdicts,
   experiments.
5. The linked issue (#46-#51) + its lane report under
   [reports/](reports/alignment-synthesis.md) — acceptance criteria,
   provenance, test plan.
6. [reports/alignment-synthesis](reports/alignment-synthesis.md) — how
   the package was checked (dedup, citations, numbers).

## Metric map pointer

The single metric → scorer → status → baseline map lives in
[metrics](metrics.md) (unified map section). Formal definitions of the
M1-M6 safety metrics stay in
[safety-metrics](safety-metrics.md); VLM derivations in the VLM note
§9.2; post-training derivations in the post-training note §12.2.
