# VLM alignment survey for footage agents (followup/alignment-vlm)

Branch: `followup/alignment-vlm`. Date: 2026-11-10. Spend: $0.00.

## What

Surveyed 21 current (2024-2026) vision-language alignment/finetuning
papers, each read in full text (arXiv HTML; search used for discovery
only), and mapped every method to HSEA footage corpora under #24
clearance. Deliverable: `docs/alignment-methods-vlm.md` (notes +
comparison table + ranked recommendations + full citations) with 3
original matplotlib figures committed under `docs/alignment-vlm/`
(rendered by `scripts/plot_alignment_vlm.py`).

## Coverage

| Family | Methods surveyed | Adopt | Adapt | Skip |
|---|---|---|---|---|
| Visual instruction-tuning mixes | Cambrian-1, LLaVA-OneVision, PixMo curation | 1 (policy) | 2 | 0 |
| Region-level / grounding | Molmo pointing, Qwen2.5-VL, PaliGemma, Griffon v2 | 1 | 2 | 1 |
| Multimodal DPO / RLHF | RLAIF-V, POVID, OPA-DPO | 2 | 1 | 0 |
| Vision RLVR | VLM-R1, Visual-RFT | 2 | 0 | 0 |
| Video-temporal | Qwen2-VL, LLaVA-Video, InternVideo2 | 0 | 1 | 2 |
| Hallucination reduction | POPE (bg), HSA-DPO, HaloQuest | 0 | 3 | 0 |
| Small-VLM / edge | SmolVLM, Bunny, FastVLM, FastV | 1 | 2 | 1 |

ID corrections made during discovery (wrong priors caught by
full-text inspection, not cited): 2410.23918 is BitStack (not
LLaVA-OneVision, real id 2408.03326); 2402.19190 is a vaccination
paper (not POVID, real id 2402.11411); 2411.05267 is channel
estimation (not SmolVLM, real id 2504.05299); 2404.14051 is an inverse
source paper (not FastV, real id 2403.06764); VLM-R1 is 2504.07615;
Visual-RFT is 2503.01785; LLaVA-Video is 2410.02713; InternVideo2 is
2403.15377.

## Recommendations (ranked)

1. Grounding + hallucination eval axis first (DisasterM3 referring,
   box-IoU/pointing, HSEA-POPE, abstain probes) — no training without it.
2. Vision-RLVR pilot on FloodNet counting + LADI severity (smallest
   compute, fixes verbose-counting failure).
3. On-policy footage DPO (RLAIF-V + OPA-DPO + POVID negatives).
4. Disciplined SFT base (Cambrian caps + ratio study, OneVision staging,
   no-distillation rule).
5. Edge after workhorses (SmolVLM SFT, guarded FastV); video training
   stays blocked (VIDI unlicensed).

## Follow-up issues (2, max allowed)

- #50 (area-eval): Footage grounding + hallucination eval —
  https://github.com/sanggusti/expert-spork/issues/50
- #51 (area-training): Vision-RLVR pilot —
  https://github.com/sanggusti/expert-spork/issues/51

Both carry repo-style sections (Summary + Acceptance criteria +
Provenance + Test plan) and reference `docs/alignment-methods-vlm.md`
§§2-4/§9 on this branch. #51's grounding-reward slice is blocked on
#50; counting/severity can start immediately. #50 complements (does
not dupe) #19 evals; #51 complements (does not dupe) #8/#9 SFT issues.

## Metrics proposed (§9 of the note)

Native inventory: REC Acc@IoU0.5, mAP, OVD recall, pointing accuracy,
counting acc/MAE, POPE F1 + Yes-rate, CHAIR/AMBER/HalBench, MHaluBench
detection, HaloQuest acc, TTFT/FLOPs, ECE. HSEA derivations with formal
definitions: `grounding_box_acc`, `pointing_acc` (+`pointing_dist`),
`hsea_pope_f1` + `pope_yes_rate`, `counting_mae_norm`, `severity_ece`,
`hal_trigger_abstain_rate`.

## Figure provenance

Original (rendered for this survey, committed PNG):
`fig1_alignment_stack.png`, `fig2_grounding_formats.png`,
`fig3_rlvr_rewards.png` in `docs/alignment-vlm/` via
`scripts/plot_alignment_vlm.py` (deterministic, Agg backend). Each was
pixel-inspected; fig2 needed two layout fixes (clipped then colliding
captions) before sign-off.
Published figures: LINK-WITH-CITATION ONLY (listed in the note §10
with paper/figure pointers). NOTHING VENDORED — including from CC-BY
papers (e.g. Visual-RFT is CC BY 4.0, FastVLM CC BY-SA 4.0, both seen
on their arXiv HTML license lines) — to keep provenance trivially
auditable. No license-cleanliness question arises.

## Verification

- `pytest tests/` green via main-repo `.venv` python against this
  worktree: **244 passed, 7 skipped in 6.58s** (`hsea` resolves to the
  worktree via PYTHONPATH); docs-only change plus one render script,
  no scorer/eval code touched.
- Repro for figures: `scripts/plot_alignment_vlm.py --out
  docs/alignment-vlm` (matplotlib 3.10.9, MPLCONFIGDIR to a writable
  dir).

```bash
cd /tmp/expert-spork-align-vlm
PYTHONPATH=/tmp/expert-spork-align-vlm \
  /Users/sanggusti/Workbench/local_development/expert-spork/.venv/bin/python \
  -m pytest tests/ -q
```

## Cost

- Estimate (pre-run): $0.00 — CPU-only docs + matplotlib figures, no
  Modal, no GPU, no API evals. Ledger TOTAL + $0 < $50 cap → GO.
- Actual: $0.00. Ledger row `50 align-vlm-survey cpu 0.5 $0.00`
  appended to `/tmp/modal-spend.log` and unioned into
  `docs/reports/modal-spend.md` (TOTAL recomputed).
- Running TOTAL after this branch: see ledger (includes parallel
  issue-1/issue-2 GPU spend, not survey spend).

## Shutdown

No Modal apps started — nothing to stop. No GPU spend. No PR opened,
no merge (per task; branch `followup/alignment-vlm` pushed for review).
