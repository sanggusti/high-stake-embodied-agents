# Alignment post-training survey — report

Branch: `followup/alignment-posttraining`. Date: 2026-11-10.
Spend: $0.00 (research + docs only, no Modal/GPU, no inference).

## What was delivered

- `docs/alignment-methods-posttraining.md` — 27 methods across 9
  families (SFT variants, DPO family, RLVR/verifiable rewards, online
  RL, process supervision + PRMs, test-time training/scaling,
  preference-data curation, multi-turn/agent RL, refusal/safety
  preserving utility). Every method has: what / when it wins / cost /
  HSEA verdict (adopt/adapt/skip) / concrete experiment on
  flywheel-v2 (SFT 8,673 / DPO 7,000 / RLVR 250) or abstention (60)
  data. Includes comparison table (§10), ordered recommendations
  (§11), native-metrics inventory + 10 formal HSEA metric definitions
  (§12), and 27 full citations with arXiv links (§13).
- `docs/alignment-methods-posttraining/` — 3 original matplotlib
  diagrams + render script + FIGURES.md provenance.
- 2 follow-up GitHub issues (labels `experiment,area-training,p2`).

All paper claims were verified against primary HTML/PDF content, never
abs pages alone (ar5iv/arxiv.org HTML for 20+ papers; PPO from PDF
text via pypdf).

## Methods covered (27)

SFT: data-centric SFT (LIMA), RFT. DPO family: DPO+length-norm, IPO,
ORPO, SimPO, KTO. RLVR: Tülu-3 RLVR, R1 rule-rewards/cold-start/
distill. Online RL: PPO (skip), GRPO, ReMax. PRM: Lightman PRM,
Math-Shepherd. Test-time: Snell compute-optimal, s1 budget forcing.
Curation: UltraFeedback, RLAIF, CAI. Agent RL: Search-R1, SWE-RL,
ToRL, Chu SFT-mem/RL-gen. Safety: Bianchi mix, Qi drift, XSTest,
WildJailbreak.

## Recommendations (leverage order)

1. E-RLVR pilot (GRPO + verifiable safety-procedure rewards) — issue #48.
2. E-DPO bake-off (length-norm DPO vs SimPO vs KTO) — issue #49.
3. Test-time verifier stack (best-of-N + PRM + XSTest contrasts) — $0 train.
4. Label flywheel (RLAIF + CAI + UltraFeedback template).
5. Safety-dose calibration (E-SAFEMIX + E-WILD).
6. Edge-lane recipe (ORPO vs R1-distill) after 27B pilots.
Non-goals: default PPO, wholesale WildJailbreak import, neural RMs
where checkers exist, more SFT cloning instead of RL.

## Issues created

| Issue | Title | URL |
|---|---|---|
| #48 | RLVR pilot: verifiable safety-procedure rewards with GRPO (E-RLVR) | https://github.com/sanggusti/expert-spork/issues/48 |
| #49 | Preference-stage bake-off: length-norm DPO vs SimPO vs KTO on flywheel-v2 pairs (E-DPO) | https://github.com/sanggusti/expert-spork/issues/49 |

## Figure provenance

| Figure | Status | License |
|---|---|---|
| `fig-rlvr-safety.png` | Original (matplotlib) | HSEA-authored |
| `fig-preference-map.png` | Original (matplotlib) | HSEA-authored |
| `fig-testtime-verify.png` | Original (matplotlib) | HSEA-authored |
| 10 published figs (DPO, SimPO, Tülu 3, R1, GRPO, Snell, s1, Search-R1, SWE-RL, WildTeaming) | Linked only | Respective arXiv perpetual — not vendored |
| XSTest Table 1, ToRL curves | Linked; eligible to vendor | CC BY 4.0 (per arXiv pages) |
| Vendored published figures | 0 | — |

Full table: `docs/alignment-methods-posttraining/FIGURES.md`.

## Cost

- Estimate (pre-run): $0 — web research + docs + local matplotlib
  render; no inference, no Modal. Live ledger TOTAL at seed-check
  $4.70, so $4.70 + $0 < $50 → GO.
- Actual: $0.00. Ledger row `48 align-pt-survey cpu 0.5 $0.00`
  (appended to `/tmp/modal-spend.log`; `modal-spend.md` updated as
  union incl. parallel agents' issue-2 rows).
- Running TOTAL after this work: **$4.70**.
