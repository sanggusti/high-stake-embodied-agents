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

## 2. Shared training recipe (document deviations)

- LoRA SFT with the same rank/alpha/target modules, same epochs/steps budget,
  same prompt-format family.
- Same alignment pass (SafeTune/DPO) on the same preference/compose set.
- Model-specific necessities only (chat template, max length, tokenizer
  quirks), each recorded in the DEVIATIONS table.
- Record: full config, GPU type, duration, cost, HF revision pin, and a model
  card (base, data rev, config).

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
