# Issue #41 — Space showcase redesign (Emberline), part 1: theme/content/assets

Branch: `issue-41`. Plan: [issue #41 plan comment](https://github.com/sanggusti/expert-spork/issues/41#issuecomment-6759520015) (§A–§G).
Part 1 (this commit): Agent A scope — theme, layout, assets, content.
Part 2 (Agent B, next): interactive examples + step replay + screenshots + final verification.

## Space push

- Space `sanggusti/high-stakes-embodied-agents`: `050f93d` → `075f9e4`
  (`Showcase redesign part 1`, theme + hero/stats/how-it-works + figure assets).
- Authenticated `/config`: HTTP 200, Gradio 6.31.0, 4 frozen labels
  (`Overview`, `Findings`, `Interactive examples`, `Inference demo`),
  theme hook ids (`showcase-hero`, `stat-strip`, `honesty-box`,
  `showcase-footer`) present in live config.

## Assets (all < 50,000 bytes; provenance in Space `assets/PROVENANCE.md`)

| File | Bytes | Source |
|---|---|---|
| `assets/hero.jpg` | 38,766 | generated (plan §C2 prompt) |
| `assets/safety.jpg` | 18,832 | generated (plan §C2 prompt) |
| `assets/loop.svg` | 1,660 | hand-authored (§B3) |
| `assets/fig-safety.jpg` | 44,967 | recompressed from `docs/reports/issue-14-safety-stack.png` |
| `assets/fig-riskcov.jpg` | 48,786 | recompressed from `docs/reports/issue-21-abstention-risk-coverage.png` |
| `assets/fig-rt-family.png` | 38,819 | copied from `docs/reports/issue-22-redteam-asr-family.png` |
| `assets/fig-curation.jpg` | 28,770 | recompressed from `docs/reports/issue-7-flywheel-v2-curation.png` |

New-binary total ≈ 221 KB (budget ≤ 300 KB). Skipped per plan:
`fig-rt-severity.png`, `fig-calibration.jpg` (gallery shows 4 max),
`docs/alignment-research/*.png` (low value, big).

## Checks (part 1)

- `py_compile` on Space `app.py`: green; `findings.json` valid (bytes unchanged).
- Tab labels frozen (4/4 in live `/config`); Interactive-examples and
  Inference-demo logic untouched.
- No Gradio-internal selectors (only the guarded underline + a comment
  mentioning them); no JS; `requirements.txt` untouched.
- Secrets grep over the diff: no tokens/keys (only benign "tokens total" copy).
- Links: §B1 allow-list only (collections/VIDI/unmerged-writeup excluded).

## Number-trace table

Full table = plan §B4 (F1–F4 footage, S1 forest_fire 0.745, T1 safety
1.000, T2 abstention risk 0.000, T3 red-team ASR 0.000, V1 flywheel
null −0.049, V2 dataset rev `b44237e1`, L1 frozen sim, D1 DORA no-go,
C1 spend $0.50). Displayed copy follows the §B4 discipline rules
(no unstated LADI counts, `fmt3()` for `0.580`).

## Deferred to Agent B

- Interactive-examples block + `video_replay.py` step replay (§D).
- Before/after screenshots (1440px + 390px) — no screenshot tooling in
  this environment; capture against the live Space with an authenticated
  session, commit next to this report.
- Full §F2 checklist sign-off, README/UPDATE_WORKFLOW replay touch-ups.

## Cost

$0.00 (no GPU, no paid inference). Ledger row `41 showcase-A cpu $0.00`;
running TOTAL unchanged.
