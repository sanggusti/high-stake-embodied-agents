# Figure generators on this branch (`figure/why-stacks`)

Two Space figure generators, both deterministic matplotlib (Agg,
DejaVu Sans, Emberline palette, arrow endpoints computed from box
edges). Bench pin for every number: origin/main
`46f35ec0e5cd64cf5e5b1f9ce204c70da8d3af17`.

## `plot_why_stacks.py` (new)

Renders the two Technical-details diagrams for the merged why-stacks
Space section (`fig-tech-why.jpg`, `fig-tech-order.jpg`, each
<50 KB via the bench PIL recipe: RGB, width 900, JPEG q82 first
rung). Numbers in the diagrams are fixed labels, not computed; each
label traces to the bench artifact below. Run:

```bash
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python \
  scripts/plot_why_stacks.py --out-dir /tmp/whyfigs
```

Per-label sources (all paths at the pin above):

- `8,673 / 7,000 / 250`: `docs/reports/issue-7-flywheel-v2-curation.md`
  (§2 split table: sft.jsonl 8,673 rows, dpo.jsonl 7,000 pairs,
  rlvr.jsonl 250 episodes).
- `base safety 0.218`: `docs/reports/issue-73-loop.md` §3
  (safety label-match base 0.218, 17/78).
- `safety 0.218 to 0.846`, `#73 SFT peak, DPO extends`:
  same §3 (SFT 0.705, abstention peak 0.650, DPO 0.846).
- `ASR 0.533 to 0.833 caught`: same §3 (red-team ASR SFT 0.533,
  16/30; DPO 0.833, 25/30) plus §4 (T4 confirms 10/11 dpo rows).
- `Gate 2 OPEN both loops`: `docs/safety-gates.md` (Gate 2 bars)
  plus `issue-73-loop.md` §12 re-grade (every arm Recover, IPO
  closest 2+2) and `issue-74-aya-expanse.md` §12 (repair Recover
  5+1, KTO 7+1).
- `T5 5-18 repaid 20-2`, `#74 DPO to repair`:
  `issue-74-aya-expanse.md` §6 (base-vs-DPO: base 18, DPO 5,
  1 inconsistent; base-vs-repair: repair 20-2 under v41).
- `26/26 overturn`: same §4 (T2 judges overturn the deterministic
  FAIL; carried refusal 26/26 both checkpoints).
- `31/40 tie`: `docs/reports/issue-62-meta-judge.md` §2 (three Meta
  configs tie 31/40 core gold-match across the $0.25-$17.83/1k
  cost spread).
- Order-chain skip costs reuse the same four labels (SFT contract,
  ASR drift, T5 tax).

Repro check 2026-11-12 (bench `.venv`, matplotlib 3.10.9,
PIL 12.3.0): branch script output matches the shipped Space bytes
exactly (`fig-tech-why.jpg` 45,508 B `275dbbc5…`, `fig-tech-order.jpg`
30,384 B `360002f1…`).

## `plot_loop_showcase.py` (rescued)

The stranded round-2 generator (originally `/tmp/plot_loop_showcase.py`;
that path is the only pre-branch copy). Rendering code is unchanged;
only the hardcoded `/tmp` input/output paths moved to `--bench-dir`
(default: repo root) and `--out-dir` (required). Its TRACE comments
cite `runs/aya73_train_log.json`, `runs/aya74/training_curves.json`,
`runs/aya73-recompute/before_after.json`, `runs/aya74/recompute.json`,
and `runs/aya74/det_scores.json`.

Repro check 2026-11-12 (same toolchain, `--bench-dir` pointed at
the round-2 `/tmp/bench-46f35ec` export): all three outputs match
the shipped Space bytes exactly (`fig-loop-loss.jpg` 45,577 B
`15fd524d…` q72, `fig-loop-stages.jpg` 43,010 B `5863d41f…` q82,
`fig-loop-xmodel.jpg` 36,579 B `79bd1ca4…` q82). Zero drift.
