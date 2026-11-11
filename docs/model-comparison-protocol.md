# Model comparison protocol: Qwen3.8-27B vs Tiny Aya vs Aya Expanse 8B

Goal: run an identical post-training + eval loop on all three models so results
compare fairly. Applies to issues #8, #73, #74. No silent divergence: every
deviation goes in a DEVIATIONS table in the issue report.

## 1. Shared data (identical revisions for all three)

- SFT/align data: `sanggusti/dmb-flywheel-v2` (pin the revision; record it).
- Held-out: the curated held-out split from #7. Same split for all three;
  never train on it.
- Safety gates: the same HarmBench + AIR-Bench refusal item lists for all
  three. Refusal must not drop vs the base model.
- Judge tracks: footage VQA/counting, safety refusal, abstention, red-team,
  dispatch-pair, per `docs/judge-playbook.md` templates T1-T5.

## 2. Shared training recipe: the full loop (document deviations)

Six stages, in order; every stage re-runs the safety stack (Qi et al.
drift rule). Research verdicts behind each choice live in
`docs/alignment-methods-posttraining.md` §10–§11; the verdict tags
below (ADOPT/ADAPT) point there.

1. **SFT (AlignTune).** QLoRA (NF4, bf16 compute) on the flywheel-v2
   SFT file: r32/alpha64 (r16 for ≤8B lanes), 3 epochs, eff-batch 8,
   lr 2e-4 cosine, seed of the issue. Format/contract teaching —
   curation beats scale (§1.1 ADOPT). No packing without prebuilt
   fused kernels; never source-build kernels.
2. **DPO (AlignTune pref).** Length-normalized DPO
   (`sigmoid_norm`, beta 0.1, 1 epoch, eff-batch 8, lr 5e-5) over the
   SFT adapter on the 7,000 dispatch pairs (§2.1 ADOPT). Bake-off arm:
   IPO on the identical config (§2.2 ADAPT, insurance vs DPO overfit
   on the templated pairs) — run it unless the issue slice forbids,
   else log the estimate and defer. (ORPO belongs to the #73 edge
   lane only, §2.3.) Pair hygiene: length-ratio debias filter (§7.1
   ADAPT); the on-policy-rejected upgrade is deferred with estimate.
3. **Drift audit.** HarmBench + AIR-Bench refusal vs base, AgentHarm
   harm-score (misuse-compliance paired with benign compliance),
   benign-utility flat check (footage + mission deltas within 0.05),
   through the #10 drift harness (`scripts/safetune_drift.py`;
   Harden/Recover/Steer). XSTest over-refusal counter-metric (§9.3
   ADOPT) pairs every refusal rate. Gate 1 is the exit bar.
4. **Repair (SafeTune).** Safety-mix re-tune (continued QLoRA, §9.1
   dose: 100s of demos) until Gate 2 passes. Mix: safety-contract
   variants + E-WILD disaster-misuse adversarial + benign lookalikes
   (§9.4 ADAPT) + abstention `cal` + SFT ballast; held-out slice per
   bench is the clean Gate-2 read. KTO (§2.5 ADOPT) is the sanctioned
   tool for binary abstention labels when pairs do not exist.
5. **RLVR/GRPO pilot (AgentTune).** GRPO (G=4, ≤256-token
   completions, rule rewards only — no neural RM, R1 pattern §3.2)
   with reward = mean of dispatch-priority match, label match,
   confirm-before-irreversible, format check (§3.1 ADOPT), plus
   SWE-RL field-match trajectory rewards where gold plans exist
   (§8.2 ADAPT) and Search-R1 observation masking on multi-turn
   rollouts (§8.1 ADOPT). Pilot-first, estimate-first, capped at
   ~40% of the issue slice; scale only if gates pass. Mandatory
   baseline: E-RFT (verifier-filtered SFT, §1.2 ADOPT) on the same
   verifier — RLVR must beat it. ReMax (§4.3) only as fallback if
   GRPO is unstable. Inference-only: best-of-N + verifier (§6.1
   ADOPT) with mean-N cost attached. Every stage reports the
   OOD rule-shift gap (§8.4 ADOPT: re-thresholded + entity-swapped
   dispatch).
6. **Evals.** The §3 battery + judge playbook, before/after tables
   per stage, storytelling plots. Replay spec: per-stage
   before/after example pairs in the report; saved bundles
   `runs/issue-N/replays/*.jsonl` (inputs, per-stage generations,
   judge reasoning+verdicts, seeds); fixed-seed
   `scripts/replay_loop.py --issue N` reproducing the comparison;
   bundles uploaded to HF.

- Model-specific necessities only (chat template, max length,
  tokenizer quirks, LoRA rank by lane), each recorded in the
  DEVIATIONS table.
- Record per stage: full config, GPU type, duration, cost, HF
  revision pin, and a model card (base, data rev, config).
- Binding GPU rules: fallback chains on every function, probe-first
  with the 2x abort rule, 80GB jobs sequenced across #8/#73/#74,
  ledger rows on the branch before the PR.

## 3. Shared eval battery (base AND finetune, all three)

- Before/after tables on the held-out split: same metrics, same columns
  (metric, base, finetune, delta).
- Safety gates: refusal non-regression vs base on the shared item lists.
- Judges: Meta `muse-wonder-1.0` with `reasoning_effort:low` as primary
  (or a recorded equivalent); reasoning + verdict, never one-word answers;
  agreement spot-checks per playbook section 4; judge spend within the
  section 5 budget rule.
- Recompute metrics with `scripts/recompute_metrics.py`; post eval ids +
  fingerprints; run the `--sanity` gold-vs-gold check.

## 4. Shared report shape

- `docs/reports/issue-N-*.md` with result TABLES and matplotlib storytelling
  PLOTS, prose between figures, same plot set across the three issues.
- The cross-model comparison table/plot lands as a follow-up after all three
  complete; it must not block any single issue.

## 5. Conformance

- At merge time each issue is checked against sections 1-4. Divergent evals
  are re-run under this protocol in a follow-up before the comparison ships.
