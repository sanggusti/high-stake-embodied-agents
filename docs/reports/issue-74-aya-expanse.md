# Aya Expanse 8B full loop: SFT + DPO + judge-backed evals (issue #74)

Status: complete. Objective: full SFT + safety-alignment + judge-backed
eval loop for Aya Expanse 8B, identical to the Qwen3.8 (#8) and Tiny Aya
(#73) loops per [model-comparison-protocol](../model-comparison-protocol.md).
No silent divergence: every deviation is in §8.

Headline: abstention exact 0.225 to 0.775, red-team ASR 0.767 to 0.367,
footage text-floor exact 0.224 to 0.755 (template-prior learning, no
vision), dispatch duels regress 5-18 to base (T5, footer leakage plus
hedged actions), safety deterministic exact dips 0.872 to 0.795 on a
footer-label artifact that T2 judges overturn (carried decisions match
gold 6/6, quality 4-5). Refusal gate: deterministic FAIL, judged PASS;
zero unsafe complies introduced.

Copy rules: no em-dashes; lanes/tracks vocabulary only; every number
traced to a log or manifest; each figure argued in prose.

## 1. Model and data

- Base: `CohereLabs/aya-expanse-8b` @
  `5062468bf9bc0c6035fd64e06274333ec127d980` (verified on the Hub,
  auto-gated, CC-BY-NC-4.0, research-only). Text-only, 8K context.
- SFT/align data: `sanggusti/dmb-flywheel-v2` @
  `07c4e27919e8ddabbb358dda3c54ea24a6e7dd66` (local mirror hash-matched
  to the Hub: sft.jsonl `5b944648...`, dpo.jsonl `a3224619...`; see
  `data/aya74/manifest.json`).
- Mix (`scripts/compose_aya74.py`, seed 74): flywheel SFT 8,673 +
  HSEA safety demos 78 + abstention demos 80 = 8,831 SFT rows;
  flywheel DPO 7,000 + safety pairs 78 = 7,078 DPO pairs. DisasterM3
  list-content rows joined (750); l2 tool trajectories flattened to
  plain transcripts (251, `[tool-call]` / `[tool result]` markers).
- Clearance: no personally-sign source per
  [license-clearance-24](../license-clearance-24.md) (the composer
  asserts this; `tests/test_aya74_compose.py` pins the counts).
- Held-out (never trained on): the #19 footage splits (FloodNet-VQA
  validation @ `ec9277af`, LADI-v2 test), `data/redteam/attacks.jsonl`
  (30), `data/dispatch-pairs/dpo.jsonl` (24). Safety (78) and
  abstention (80) rows ARE trained on for contract teaching; those
  lanes report as contract checks wherever they appear.

## 2. Training recipe (shared protocol §2)

- Stage 1 SFT: LoRA r16/alpha32/dropout 0.05, all 7 projections,
  3 epochs (3,312 steps), effective batch 8 (2 x 4), LR 2e-4 cosine,
  warmup 5, max_length 4096, packing off, bf16, seed 74.
  Final train loss 0.371 (`runs/aya74/dl/sft_log_history.json`).
- Stage 2 alignment: DPO sigmoid beta 0.1 over the merged SFT
  checkpoint (reference = merged SFT, frozen), 1 epoch (885 steps),
  effective batch 8 (1 x 8), LR 5e-5 cosine, warmup 20, max_length
  2048. Final train loss 0.008 (`runs/aya74/dl/dpo_log_history.json`).
- Infra: Modal A100-80GB for SFT+DPO, A10G merge, A10G generation.
  vLLM serving was deployed twice (L4, then A10G) but its HTTPS
  tunnel never returned a body, even probed from inside Modal's
  network; evals ran via direct Modal generation instead (D5).

![Training loss curves](issue-74-loss-curves.png)

SFT loss falls 3.17 to 0.26 over 3,312 steps with a healthy noise
floor. DPO loss collapses 0.63 to ~0 by step 100 and stays there:
the pairs separate almost perfectly, which is an overfit warning,
not a triumph. The dispatch regression in §6 reads as its price:
with nothing left to learn from the pair loss, the checkpoint keeps
the SFT contract style (footers, deferral) while losing action
concreteness. Source: Modal trainer log histories plotted by
`scripts/aya74_eval.py plot` from `runs/aya74/training_curves.json`.

## 3. Before/after tables (protocol §3)

Source: `runs/aya74/recompute.json` (`scripts/aya74_eval.py
recompute`; key-paired base vs finetune=DPO, McNemar exact two-sided,
paired bootstrap deltas from `hsea.stats`). Finetune = DPO adapter;
the SFT intermediate appears in §4 prose and plots.

| metric | base | finetune | delta | n | mcnemar p |
|---|---|---|---|---|---|
| safety exact | 0.872 | 0.795 | -0.077 | 78 | 0.0703 |
| refusal on HarmBench+AIR-Bench | 1.000 | 0.885 | -0.115 | 26 | 0.2500 |
| benign comply | 1.000 | 1.000 | +0.000 | 10 | 1.0000 |
| abstention exact | 0.225 | 0.775 | +0.550 | 80 | 0.0000 |
| abstention soft | 0.225 | 0.844 | +0.619 | 80 | 0.0000 |
| red-team ASR (lower better) | 0.767 | 0.367 | -0.400 | 30 | 0.0042 |
| footage text-floor exact | 0.224 | 0.755 | +0.531 | 49 | n/a (unpaired lane) |

![Deterministic lanes](issue-74-det-lanes.png)

Three lanes move strongly and significantly (abstention, red-team,
footage floor), one holds exactly (benign comply 10/10 both), and
one dips without significance (safety exact, p = 0.07). The SFT
intermediate (orange) matches or beats DPO (green) on abstention
(0.863 vs 0.775) and red-team (0.333 vs 0.367): the alignment pass
adds nothing measurable on these lanes and costs a little, while
DPO leads only on the footage floor (0.755 vs 0.714). Whiskers are
Wilson 95% (`runs/aya74/det_scores.json`).

## 4. Safety lane and the refusal gate

Deterministic gate: FAIL (refusal 1.000 to 0.885, delta -0.115,
p = 0.25, n = 26). The six flipped rows are a footer-label
artifact, not a behavior regression: four refuse-gold rows carry
refusal text ("I can't help... policy forbids") with a
`DECISION: abstain` footer, and two escalate-gold rows carry
escalation text (one literally names the incident commander gate)
with an abstain footer. Zero flips go toward comply on harmful
rows. Cause: the SFT mix teaches two footer vocabularies (4-way
DECISION plus 3-way ACTION) and the checkpoint over-applies
abstain; follow-up: unify the footer contract.

T2 judges overturn the FAIL on carried decisions: all six flips
grade as the gold decision by both judges (refuse rows quality
4-5, escalate rows quality 4-5; `runs/aya74/judgments.jsonl`, keys
`T2:*`). Carried refusal on HarmBench+AIR-Bench is 26/26 for both
checkpoints. Verdict: deterministic FAIL (footer artifact, kept
visible), judged PASS (behavior held).

![Safety Pareto](issue-74-safety-pareto.png)

Benign utility holds at 1.0 for all three checkpoints while
deterministic refusal slips 0.95 to 0.875 on the artifact above;
no point trades safety for capability or the reverse. Points are
(base 38/40 refuse, sft 36/40, dpo 35/40) over the 40 refuse rows
plus 22 benign rows (`runs/aya74/det_scores.json`).

## 5. Abstention, red-team, footage lanes

Abstention (contract check, trained on): exact 0.225 to 0.775,
soft 0.225 to 0.844, both p < 0.0001. Base Aya rarely emits the
ACTION footer; the loop teaches the contract. T3 dual agreement
31/36, kappa 0.78, bar pass.

Red-team (held out): keyword ASR 0.767 to 0.367 (p = 0.0042),
with per-family ASR falling in jailbreak (9 to 4 of 10), misuse
(7 to 5 of 10), and retrieval_injection (7 to 2 of 10). T4
adjudicates the 8-row dpo review queue as 5 succeeded / 3 refused
against 7/1 for base on the same keys: directionally better, too
small to claim. T4 dual agreement 14/16, kappa 0.73, bar pass.
The keyword ASR stays the regression gate per the playbook.

Footage (held out, text floor): exact 0.224 to 0.755. This is
template-prior plus format learning, not vision: yes/no and
condition questions are 90%+ skewed per template in train (e.g.
"Is the entire road non flooded?" is yes 48/49 in the 1,000-row
train slice), and the checkpoint learns P(answer|question) plus
the single-word contract. Counting stays near chance (3/5 dpo,
train answers near-uniform). Different images from train (train
vs validation/test splits), so no leakage; the floor measures
priors honestly. T1 dual agreement 26/30, kappa 0.76, bar pass;
on the 15-row dpo-missed sample the judges rescue 3 correct + 2
close, while base completions on the same keys rescue 10 correct,
showing the misses are dpo-specific, not question hardness.

## 6. Dispatch duels (T5): the regression

T5 base-vs-DPO duels over the 24 held-out dispatch situations,
swap-twice, judge A: base wins 18, DPO wins 5, 1 inconsistent
(conflict rate 4.2%). DPO win rate 0.208 (Wilson 0.09 to 0.40).

![T5 duels](issue-74-t5-duels.png)

The chart needs no suspense: base wins nearly four to one. Two
observed mechanisms, both traced to completions. First, footer
leakage: DPO dispatch rows carry contract footers with off-label
values (`ACTION: standdown`, `ACTION: notify_coordinator`,
`DECISION: prioritize`, plus `draft`, `assign`, `the`) in a
setting where no footer exists. Second, hedged actions: where
base writes concrete dispatch steps, DPO downgrades (P1 to P2,
P2 to P3 on sampled rows) or proposes notifying before acting,
and the T5 priority-first rubric punishes exactly that (e.g.
`T5:orig:dispatch:c-01`, margin 3: both P1, base wins on action
concreteness). This is the alignment tax made visible: the
safety/abstention contract generalized into the dispatch head as
deferral. Follow-up: footer-vocab unification plus dispatch-pair
rebalancing before any release claim.

## 7. Judge agreement (§4 bars)

Dual judges from different families on every judged row (DeepSeek
+ Gemma via the HF router, the recorded equivalent; D3). Bar:
agreement at least 0.80 AND kappa at least 0.60, else a third
judge. All four single tracks pass, so no third judge ran.

| track | n | dual agree | kappa | bar |
|---|---|---|---|---|
| T2 safety | 44 | 41 (0.93) | 0.90 | pass |
| T3 abstention | 36 | 31 (0.86) | 0.78 | pass |
| T4 red-team | 16 | 14 (0.88) | 0.73 | pass |
| T1 footage | 30 | 26 (0.87) | 0.76 | pass |

Spend: 348 calls, 472,763 in / 50,348 out tokens, $0.04 reported,
29.0 min (`runs/aya74/judge_spend.json`), within the §5 $2.00 cap
after estimate-first ($0.40 upper bound). T1 ran a seeded 12-row
zero-sample per §5 rule 4 (shrink the sample, never the protocol).

## 8. DEVIATIONS from the shared protocol

| # | Area | Protocol | This issue | Reason |
|---|---|---|---|---|
| D1 | DPO max length | same recipe | 2048 (SFT 4096 as shared) | dual-model DPO fit on one 80GB GPU; allowed model-specific max length |
| D2 | footage lane | VQA/counting on held-out | text floor with [frame omitted] | Aya Expanse 8B is text-only; cannot take pixels (same class as Tiny Aya) |
| D3 | judges | Meta muse-wonder-1.0 primary | HF dual pair (DeepSeek + Gemma) | no Meta API key in this environment; the recorded equivalent, §4 dual-family bar met |
| D4 | SFT seed | unspecified | 74 | issue seed, recorded |
| D5 | eval transport | inspect-ai over vLLM + .eval logs | direct Modal generation, key-paired recompute | vLLM HTTPS tunnel never returned a body (L4 and A10G, also probed from inside Modal); identical prompts, systems, sampling (temp 0), and scorer functions; follow-up can re-run under inspect |
| D6 | T1 sample | triggered rows | seeded 12-row zero-sample + audit | playbook §5 rule 4 budgeting; recorded |

Safety/abstention train overlap is disclosure, not deviation: the
issue composes from those sets and the protocol only protects the
#7 held-out split.

## 9. Artifacts and spend

- HF (owned by this issue): `sanggusti/aya-expanse-8b-hsea-sft`
  (LoRA SFT adapter + tokenizer + trainer state + model card) and
  `sanggusti/aya-expanse-8b-hsea-dpo` (LoRA DPO adapter + tokenizer
  + trainer state + model card).
- Eval ids: `eval_base.jsonl`, `eval_dpo.jsonl` (+ `eval_sft.jsonl`
  intermediate), prompts `eval_prompts.jsonl`, judgments
  `judgments.jsonl`, all under `runs/aya74/`.
- Fingerprints: `runs/aya74/recompute.json` trace (upstream
  revisions + local sha16 of every eval input).
- Sanity: `scripts/recompute_metrics.py --sanity` → SANITY OK
  (footage 4, safety 78, abstention 80, redteam 30).
- Tests: 307 passed, 7 env-skipped (full suite), including
  `tests/test_aya74_compose.py` (mix pins) and
  `tests/test_aya74_eval.py` (plan pairing, T5 math, recompute).
- Spend (ledger rates: a100-80gb $0.0416/min, a10g $0.0185/min,
  l4 $0.0134/min; SFT minutes throughput-derived at the measured
  smoke rate, wall 503 min incl. unbilled queue):

| run | class | min | $ |
|---|---|---|---|
| verify/smokes CPU | cpu | 0 | 0.00 |
| sft-smoke | a100-80gb | 4 | 0.17 |
| dpo-smoke | a100-80gb | 5 | 0.21 |
| sft-full | a100-80gb | 150 | 6.24 |
| dpo-full | a100-80gb | 68 | 2.83 |
| merge | a10g | 8 | 0.15 |
| gen base/sft/dpo | a10g | 81 | 1.50 |
| gen footage x3 | a10g | 12 | 0.22 |
| serve boots | l4/a10g | 15 | 0.24 |
| judges | api | 29.0 | 0.04 |
| slice total | | | 11.60 |

Slice $11.60 (cap $25). Every Modal app stopped (0 tasks); ledger
rows land in `docs/reports/modal-spend.md` with this branch.

## 10. Follow-ups

1. Footer-vocab unification (DECISION vs ACTION, off-label values)
   plus dispatch-pair rebalancing; re-run T5 to clear the tax.
2. Re-run evals under inspect-ai + vLLM when the tunnel cooperates
   (D5 travel note for the cross-model comparison).
3. DPO early-stop or lower LR: loss hits ~0 by step 100 of 885.
