# Alignment synthesis: unified index + dedup audit

Branch: `followup/alignment-index` (worktree
`/tmp/expert-spork-align-index`, from `origin/main`). Date: 2026-11-10.
Spend: $0.00 (research + docs only, no Modal/GPU, no inference).
Main checkout untouched; lane branches read via `git show` only, never
merged or edited.

## What was delivered

- `docs/alignment-methods.md` — unified index: one paragraph per lane,
  ranked cross-lane recommendation table with effort/impact/cost,
  reading order for future agents.
- `docs/metrics.md` — new "Unified metrics map" section: every metric
  → scorer file → status (implemented / proposed) → baseline value
  with report trace. Formal M1-M6 definitions stay in the safety lane's
  `docs/safety-metrics.md`; VLM derivations in the VLM note §9.2;
  post-training derivations in the PT note §12.2.
- `docs/README.md` — index rows for all five package docs (index + 3
  lane notes + safety-metrics) plus the metrics-map description bump.
- This report — coverage map, ranking rationale, dedup findings,
  citation spot-check log, HSEA number audit.

Lane sources (all read-only): `followup/alignment-posttraining`
(`docs/alignment-methods-posttraining.md`, issues #48/#49),
`followup/alignment-vlm` (`docs/alignment-methods-vlm.md`, issues
#50/#51), `followup/alignment-safety`
(`docs/alignment-methods-safety.md` + `docs/safety-metrics.md`, issues
#46/#47).

## Coverage map

| Family | Post-training lane | VLM lane | Safety lane |
|---|---|---|---|
| SFT / data curation | LIMA/RFT, UltraFeedback, RLAIF, CAI | Cambrian caps, OneVision staging, PixMo policy | — (training-side, deferred) |
| Preference / DPO | length-norm DPO, IPO, ORPO, SimPO, KTO (#49) | RLAIF-V, POVID, OPA-DPO (no issue filed) | SecAlign cited, deferred to AgentTune |
| RLVR / verifiable rewards | safety-procedure RLVR + GRPO (#48) | counting/severity/grounding RLVR (#51) | — |
| Process supervision / test-time | PRM rerank, Math-Shepherd, Snell BoN, s1 forcing | confidence-calibration reward (V-RFT) | trace features (diagnostic only) |
| Refusal / over-refusal | Bianchi mix, Qi drift, XSTest, WildJailbreak pipe | HSA-DPO severity, HaloQuest triggers | XSTest/OR-Bench/SORRY, benign-hard-v2 |
| Calibration / abstention | escalate-trigger ECE (E-PRM) | severity ECE, abstain probes | SW-ECE, deferral policy, confidence spread (#47) |
| Red-team / robustness | — | — | graded ASR, R-CONS, injection-utility (#46) |
| Agent / multi-turn | Search-R1, SWE-RL, ToRL, Chu ordering | multi-frame probes | R-Judge risk-ID, AgentDojo triple, CaMeL |
| Metrics | 10 PT derivations (§12.2) | 7 VLM derivations (§9.2) | inventory + M1-M6 |
| Process / reporting | — | — | Pareto tracking (safetywashing) |

Deliberate splits, not gaps: text RLVR (#48) vs vision RLVR (#51) by
modality; graded-ASR grades (#46) vs EHWR risk (#47) by scorer;
text-DPO bake-off (#49) filed while footage DPO stays a recommendation
(each lane filed max 2 issues). A future footage-DPO issue should cite
#49's winner as its prior.

## Cross-lane ranking (rationale)

Top 3 (full table in the index):

1. **Eval tranche (#46 + #47 + #50).** Unanimous across lanes: no
   training claim without meters. Fixes three documented pathologies:
   binary ASR overstating harm (#22 mu-01), degenerate flat-0 risk
   (#21), and the missing grounding axis (footage-evals). $0 GPU.
2. **RLVR pilots (#48 + #51).** Highest-leverage training in both
   lanes, one shared GRPO + deterministic-verifier recipe, disjoint
   reward sets (safety-procedure vs counting/severity/grounding).
   Falsifiable against RFT/SFT controls. Run in parallel once rank 1
   lands; #51's grounding slice waits on #50.
3. **Preference default (#49).** One config × four losses settles the
   AlignTune default for all lanes and feeds #48's ablations. Text
   only; footage DPO follows later with the winner as prior.

Then: $0 wins immediately (rank 4), label flywheel feeding ranks 2-3
(rank 5), safety-dose calibration pre-release (rank 6), deliberative
pilot in parallel (rank 7), eval-data growth ongoing (rank 8), edge
lane last (rank 9).

## Dedup findings

All 6 issues cross-checked pairwise (titles + bodies + acceptance
criteria + provenance). Verdict: **0 true duplicates, 0 comments
posted** (nothing met the bar for a close-as-duplicate proposal).

| Pair | Overlap surface | Verdict |
|---|---|---|
| #48 vs #51 (RLVR/GRPO) | same algorithm family | NOT dupes: text safety-procedure rewards on Qwen3.8-27B vs vision counting/severity/grounding rewards on Qwen3-VL-8B; disjoint verifiers, models, evals. Share infra learnings. |
| #49 vs VLM DPO plan | preference tuning | NO overlap today: no footage-DPO issue exists; #49 is text-only. Future footage issue cites #49. |
| #46 vs #47 | M1 concept shared | Complementary by design: #46 ships harm grades + R-CONS in `redteam.py`; #47 ships EHWR/WSR/F2/SW-ECE in `abstention.py`/`metrics.py`. |
| #50 vs #47 | abstain probes | Touchpoint, not overlap: #50's HaloQuest slice wires into abstention coverage; different harnesses (footage vs text). Coordinate on the shared `hal_trigger_abstain_rate` name. |
| #48 vs #49 | same data/checkpoints | Sequential by design: #49's winner feeds #48. |
| #50 vs #51 | grounding rewards | Explicitly blocked slice (#51 waits on #50 for IoU rewards only). |
| #46/#47 vs #48-#51 | gates | No overlap: eval/metering vs training pilots; pilots consume the meters. |

## Citation spot-check log

2 arXiv URLs per lane note, fetched 2026-11-10 (HTTP 200 + title
match required). 6/6 pass.

| Lane | URL | Result |
|---|---|---|
| PT | https://arxiv.org/abs/2411.15124 | 200 — Tulu 3: Pushing Frontiers in Open Language Model Post-Training |
| PT | https://arxiv.org/abs/2501.12948 | 200 — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL |
| VLM | https://arxiv.org/abs/2409.17146 | 200 — Molmo and PixMo |
| VLM | https://arxiv.org/abs/2501.09695 | 200 — Mitigating Hallucinations in LVLMs via DPO: On-Policy Data Hold the Key |
| Safety | https://arxiv.org/abs/2412.16339 | 200 — Deliberative Alignment |
| Safety | https://arxiv.org/abs/2402.10260 | 200 — A StrongREJECT for Empty Jailbreaks |

## HSEA number audit

Every HSEA-specific number in the three lane notes + safety-metrics.md
was traced to `docs/reports/` (or the cited source file).
**Violations: none.** Trace table (claim → source):

- flywheel-v2 SFT 8,673 / DPO 7,000 / RLVR 250 → issue-7 report table
  (exact match, incl. SHA prefixes).
- abstention "60 scenarios" → issue-21 report: 60 risk scenarios (20
  abstain / 20 escalate / 20 comply) + 20 benign; cal 28 / eval 52.
- FloodNet-VQA 4.5k pairs / DisasterM3 123k pairs / D-Fire 21.5k /
  FLAME 2 53k pairs / AIDERv2 16.7k / DMD 5.9k →
  `docs/datasets.md` (4,511 / 123k / 21,527 / 53,451 / 16,723 /
  5,878 — all match).
- FloodNet frames 3000×4000 → `docs/datasets.md`; eval downsample
  1024px → `hsea/data/footage.py:png_bytes(max_side=1024)`.
- LADI 12 yes/no questions → `docs/footage-evals.md`.
- #14 stack: refusal 1.000, escal 0.667/1.000, abstain 1.000/0.833,
  agentharm benign 1.000/0.875, ECE 0.016/0.021 → issue-14 report
  table (exact).
- Reasoning tokens R:61,767 / O:84,390 → issue-14 report run row
  (exact).
- #21 eval: risk 0.000, coverage 0.481/0.423, over-refusal
  0.038/0.154, soft 0.981/0.875, ECE 0.036/0.075, MCE 0.105/0.229,
  AURC 0.000, τ* 0.00 → issue-21 report tables (exact).
- #22: v41 ASR 0.000 automated + adjudicated, severity 0.700/0.667/
  0.625 mockllm, flag rate 0.400, 30 attacks (10 jailbreak = 5 GCG +
  5 PAIR / 10 injection / 10 misuse), mu-01 partial → issue-22
  report (exact).
- #10 drift tolerances + Harden/Steer/Recover bands → safety-lane
  report (branch `origin/issue-10`, read via lane report).

Two paraphrases noted, neither a violation: "confidence saturates at
95–100" (report bins are (0.8,1.0] with means 0.855–0.970 —
approximate, directionally exact); M2/M5 "directional" backfills are
explicitly labeled as needing the implementing PR.

## Merge-order note

Lane branches merge first (each with its own PR + review); this index
branch merges after, so its lane-doc links resolve. README rows added
here for lane docs are forward links until then. A serving workflow
runs in parallel — this branch touches docs only (3 edited/created
prose docs + this report + ledger mirror), no code, no shared state
beyond the union-resolved spend mirror.

## Verification

- Full `pytest tests/` green via main-repo `.venv` python (cwd in
  this worktree; main checkout untouched): **244 passed, 7 skipped**.
- No secrets printed; no Modal/GPU/API calls.

```bash
cd /tmp/expert-spork-align-index
PYTHONPATH=/tmp/expert-spork-align-index \
  /Users/sanggusti/Workbench/local_development/expert-spork/.venv/bin/python \
  -m pytest tests/ -q
```

## Cost

- Estimate (pre-run): $0 — docs + web reads + local pytest; ledger
  TOTAL + $0 < $50 cap → GO.
- Actual: $0.00. Ledger row `51 align-index-synth cpu 0.5 $0.00`
  appended to `/tmp/modal-spend.log`; `docs/reports/modal-spend.md`
  updated as union (no rows removed, TOTAL recomputed).
- Running TOTAL after this work: see ledger (parallel serving/GPU
  spend excluded from this $0 claim).
