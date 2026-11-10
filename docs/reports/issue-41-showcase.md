# Issue #41 — Space showcase redesign (Emberline), parts 1+2 (A+B)

Branch: `issue-41`. Date: 2026-11-10. Plan: [issue #41 plan comment](https://github.com/sanggusti/expert-spork/issues/41#issuecomment-6759520015) (§A–§G).
Part 1 (Agent A): Space `theme.css`, `assets/*`, `app.py` header/hero/stat-strip/how-it-works/honesty/footer + Overview + Findings tabs + `css=` + `fmt3()` + chart recolor, README showcase §, UPDATE_WORKFLOW asset-budget + screenshot steps.
Part 2 (Agent B): `video_replay.py` + Tab 3 Radio/gallery/replay + `SAMPLE_FORMAT.md` contract + docs touch-ups + verification + live screenshots.
Render-QA (A, on top of B): single-panel stat strip, adaptive header, clean hero art.

## Space commits

- `050f93d` (baseline) → `075f9e4` (#41-A part 1: Emberline first pass) →
  `66528d6` (#41-B part 2: Agent B step-replay, built on the part-1 snapshot) →
  **`86ed7f8`** (#41-A render-QA fixes, current HEAD).
- Agent A rebased the render-QA fixes onto `66528d6` (clean, no overlap:
  B touched the `video_replay` import + Tab 3 only) and pushed `86ed7f8`.
  Agent B verified Tab 3/replay intact post-rebase (py_compile green,
  determinism PASS, live `/config` replay components present).
- Live state (`86ed7f8` RUNNING): `/config` HTTP 200, 117,660 bytes,
  Gradio 6.31.0, tabs exactly `Overview` / `Findings` /
  `Interactive examples` / `Inference demo`; theme hooks present
  (`showcase-header`, `showcase-links`, `showcase-hero`, `honesty-box`,
  `showcase-footer`, `.hero-art`, `.section-card`, `.safety-emblem`,
  `.loop-diagram`, plus B's `.replay-slot`/`.empty-slot`; `#stat-strip`/
  `.stat-card` are inner HTML by design); Timer ×1, Radio `Mode`
  (Run gallery / Step replay), Slider `Step` 0..21, empty-state copy present.
  (Earlier B check on `66528d6`: 121,834 bytes, same components; QA shrank
  config via single-panel strip.) Hardware `cpu-basic` (runtime API
  `current`/`requested`); variables `{}` (no Space secrets).

## Agent B scope (§E, Tab 3 only; zero new binaries)

- `video_replay.py` (new, dependency-free): `load_steps`
  (`trajectory_excerpt.jsonl` → `{t, step, kind, summary}` + passthrough),
  pure `get_step` (index+clamp), `cumulative_counts`, `format_step_card`
  (kind-colored note/action/edge, inline styles only),
  `format_cumulative_line`, `format_precomputed_panel` (VQA chip + running,
  or grid verbatim ≤400ch), `find_visual_samples` + `load_visual_steps`
  (#40 contract).
- `app.py` Tab 3 only (+1 import): `Radio` gallery/replay + visibility
  toggle, gallery meta +3 lines (license/provenance ≤200ch, eval log,
  detail k=v), #40 placeholder note, replay stepper (`Slider` 0..21 +
  Prev/Next + Play/Pause via `gr.Timer(0.6)` + step card + cumulative +
  precomputed panel), visual slot (§B1 empty-state; Gallery when `frames/`
  lands). Other tabs untouched (diff-verified); labels frozen;
  `theme.css`/`requirements.txt` untouched by B.
- `samples/SAMPLE_FORMAT.md`: §"Step-replay contract" appendix
  (`frames/` + `steps.jsonl` schema, budgets, license rules).
- `README.md`/`UPDATE_WORKFLOW.md`: replay touch-ups only.

Content = §B exact copy; text-mode over
`samples/hf-forestfire-v41flash/trajectory_excerpt.jsonl` (22 lines).

## Number-trace table (plan §B4 — every showcased number)

Full eval-log pins (the plan's `…`-shorthand expanded from
`findings.json` + bench reports; all re-verified 2026-11-10).

| ID | Number as shown | Report (origin/main) | Eval id / pin |
|---|---|---|---|
| F1 | 0.500±0.527 (5/10), vqa 0.500, count 0.500 | docs/footage-evals.md; #16 | `2026-11-09T12-54-34-00-00_floodnet-vqa_VJtwMHzZkg3cgsksqiyhsN.eval`, run `footage-floodnet-baseten-01` |
| F2 | 0.875±0.338, vqa 1.000, sev 0.750 (n=24) | docs/footage-evals.md | `2026-11-09T12-55-04-00-00_ladi-v2_iFkjvdUJSMxAeXjqUrtSo4.eval`, run `footage-ladi-baseten-01` |
| F3 | 0.300±0.483, vqa 0.250, count 0.500 (n=10) | docs/footage-evals.md | `2026-11-09T13-55-37-00-00_floodnet-vqa_SBB6UyhZUsba8rpAg6ZCjz.eval`, run `footage-floodnet-modal-01` |
| F4 | 0.917±0.282, vqa 1.000, sev 0.833 (n=24) | docs/footage-evals.md | `2026-11-09T13-56-15-00-00_ladi-v2_kDnU62h5JbrwcCAnEV4EBj.eval`, run `footage-ladi-modal-01` |
| S1 | mission 0.745, recall 1.0 = 6/6, 27 calls, 187,192 tok, 1:59 | #6 report | `2026-11-09T15-17-03-00-00_forest-fire_9V4WeMzVWrE7Eb8sqwpVVt.eval`, run `hf-ff1-v41flash` |
| T1 | 78 items; refusal 1.000 ×4; ECE 0.016/0.021; 4 misses safe-direction | #14 report | `2026-11-09T16-35-53-00-00_safety_caDjQESKTMY37FJzB2gsz5.eval` + `2026-11-09T16-36-15-00-00_safety_kJoUXQWapFiCSQ3oEeH2qx.eval` |
| T2 | risk 0.000 ∀τ; τ\*=0.00; soft 0.981/0.875; ECE 0.036/0.075; AURC 0.000 (n=52) | #21 report | `2026-11-09T16-23-47-00-00_abstention_njuR2QVGw4VUSJvXPNkLBZ.eval` + `2026-11-09T16-22-40-00-00_abstention_2FX4ZbVYAU4ED6zNZuSz7X.eval` |
| T3 | ASR 0.000/0.000 (mock 0.667); 30 attacks; flag 0.400; mu-01 3/4 | #22 report | `2026-11-09T16-50-54-00-00_redteam_6je68c9FmTxJkKRvZWdFFQ.eval` + `2026-11-09T16-50-56-00-00_redteam_Lpa4K4BWN4rHKScmCu9VuM.eval` |
| V1 | teacher 0.587 (30/40≥0.5) → 224 pairs; 0.580±0.158 → 0.531; Δ −0.049; 6/20 @ 0.300 | docs/flywheel.md | runs/flytrain-v1, flytest-base, flytest-ft (local-only) |
| V2 | 8,673 + 7,000 + 250 rows; 10.0/5.9/0.2 MB; rev `b44237e1…` gated | #7 report | HF `sanggusti/dmb-flywheel-v2` rev `b44237e145af1f0d9cfa3f22ec36f77d22ca5393` |
| L1 | sim legacy 6/93 LLM (6 victims), greedy 10/12 | findings.json (frozen) | #79/#80/#82 batches (retired #13); shown in headline + "Retired sim lineage" accordion |
| D1 | DORA gated:auto 403; 17.6 GB; 791 files; 10-task frozen plan | #23 report | rev `76897d47cc2e2baa46b7657ad9c707e21ff707d6` — NOT displayed in the Space (context only, no scores exist) |
| C1 | total spend $0.50 (footer) | docs/reports/modal-spend.md | ledger rows #6/#21/#14/#22 (HF upper-bound rates) |

Discipline (plan §B4, verified in `app.py`): no LADI counts (21/24,
22/24) and no Qwen 3/10 printed anywhere; `findings.json` `0.58`
displays as `0.580` via `fmt3()` (JSON bytes unchanged).

New rows (part-2 Tab 3 numbers; Agent A files add none beyond §B4):

| ID | Number as shown | Source |
|---|---|---|
| R1 | 22 lines, Slider 0..21, Step i/21, tool calls 0..11, finds 0..3, transcript coords/obs verbatim | Space `samples/hf-forestfire-v41flash/trajectory_excerpt.jsonl` (22 lines); excerpt of S1 run `hf-ff1-v41flash` (full 53 events, 27 calls per `excerpt_end`). Excerpt counts are the visible subset; full totals trace to S1 eval. |
| R2 | `detail`: rescue_recall=1.0, containment=0.35, efficiency=0.666…, safety=0.5 (hf-forestfire); metric string (footage); — (sim-era) | Space `samples/*/meta.json` `score_detail` (harvested from eval logs). Recall = S1; containment/efficiency/safety from S1 `mission_score` vector (same S1 eval log). |
| R3 | Visual empty-state: 27 calls, first find at step 3, 6/6 rescue (§B1 verbatim) | 27 + 6/6 trace to S1. "Step 3" is §B1/meta-notes verbatim; excerpt first shows novelty at t=11 (Step 11/21). Both displayed honestly; discrepancy noted. |

`renders_source`/`eval_log` are provenance IDs (traces themselves).
Doc policy thresholds (<50 KB, ≤250 KB, ≤400ch, ≥0.5) come from plan
§§C/D, not results.

## Assets shipped (Space `assets/`)

Bench source pin: `origin/main` `7f46a1ca9e4a32d1350268fdae40320295327a57`.

| Space file | Source / action | Dims | Bytes |
|---|---|---|---|
| `fig-rt-family.png` | #22 report PNG, copied as-is | 1050x600 | 38,819 |
| `fig-safety.jpg` | #14 PNG (84,308 B) → JPEG q82, width 900 | 900x332 | 44,967 |
| `fig-riskcov.jpg` | #21 risk-coverage PNG (57,639 B) → same recipe | 900x643 | 48,786 |
| `fig-curation.jpg` | #7 PNG (67,974 B) → same recipe | 900x252 | 28,770 |
| `hero.jpg` | generated (flat-vector dusk scene), 1600x500 q72 | 1600x500 | 38,766 |
| `safety.jpg` | generated (shield/drone emblem), 900x420 q80 | 900x420 | 18,832 |
| `loop.svg` | hand-authored flywheel diagram (~40 lines) | vector | 1,660 |
| `PROVENANCE.md` | per-file source + SHA-256 + prompts | — | — |

Every binary <50,000 bytes; new-binary total 218,940 bytes ≤ 300 KB
(part 2 adds zero binaries). Full SHA-256 + prompts in Space
`assets/PROVENANCE.md`. Bench-only overflow (NOT shipped):
`issue-16-space-refresh.png`, `issue-22-redteam-asr-severity.png`,
`issue-21-abstention-calibration.png`.

## Screenshots (before/after local + live interactive)

Before/after (Agent A): local `demo.launch()` at Space `86ed7f8`
(post-rebase tree) + baseline `050f93d`, `?__theme=light`, Overview tab.

| | 1440px desktop | 390px narrow |
|---|---|---|
| before (`050f93d`) | ![before desktop](issue-41-showcase-before-1440.png) | ![before narrow](issue-41-showcase-before-390.png) |
| after (`86ed7f8`) | ![after desktop](issue-41-showcase-after-1440.png) | ![after narrow](issue-41-showcase-after-390.png) |

Render notes (A): hero legible, strip 6-across → single column,
art hidden on narrow (copy-first), no added overflow. Dark-theme QA
readable (adaptive header; light shots are reference). Narrow
right-edge clipping identical in before/after: pre-existing Gradio
tab-bar behavior (4 frozen labels ≈ 550px min-width; Gradio-internal,
out of scope), not a regression. Findings-tab shots not taken
(headless cannot click tabs in that harness); verified via DOM + `/config`.

Live interactive (Agent B): Space `86ed7f8`, authenticated session,
Gradio 6.31.0, Chromium headless (Playwright, `?__theme=light`).
Tab clicks at desktop width only (narrow tab bar overflows; narrow
shots via resize-after-navigate). All 6 committed next to this report.

| File | Viewport | Tab / mode | Notes |
|---|---|---|---|
| `issue-41-overview-1440.png` | 1440px | Overview | Hero legible, 6-card strip, no overflow |
| `issue-41-overview-390.png` | 390px | Overview | Copy-first, strip stacked, no overflow |
| `issue-41-gallery-1440.png` | 1440px | Interactive / Run gallery | Radio, dropdown, 3 imgs, enriched meta + #40 note |
| `issue-41-gallery-390.png` | 390px | Interactive / Run gallery | Narrow gallery, no overflow |
| `issue-41-replay-1440.png` | 1440px | Interactive / Step replay | Slider 0..21, Prev/Next/Play/Pause, Step 0/21 card, cumulative, precomputed panel, §B1 empty-state |
| `issue-41-replay-390.png` | 390px | Interactive / Step replay | Controls stack vertically, no overflow |

Gallery-vs-replay is the meaningful before/after within Tab 3 (Radio
toggle). Verified pixels: replay "Step 0/21 · note", cumulative "0 · 0",
empty-state "27 tool calls, first find at step 3, full 6/6 rescue"
verbatim. Tooling: `playwright` + `chromium` to `/tmp/ms-playwright`
(`PLAYWRIGHT_BROWSERS_PATH`); Chromium launch needed one-time
unsandboxed approval (managed macOS sandbox blocks Mach ports); live
auth via `Authorization: Bearer` header (token from bench `.env`,
never printed).

## Render-QA deviations from plan (all deliberate, all documented)

1. Stat strip = ONE `gr.HTML` panel (`stat_strip_html()`) with an inner
   `#stat-strip` div, not 6 components in a `gr.Row`: Gradio 6 wraps
   each HTML component in its own bordered box, which double-framed
   every card. Same hooks, same CSS, cleaner render.
2. Header/tagline/vintage/links inherit Gradio's adaptive text color
   (opacity for hierarchy) instead of forcing `--ink`/`--muted`:
   forced dark text was unreadable on dark viewer themes.
3. Hero + safety images pass `buttons=[]`: Gradio 6 overlays a
   download/share/fullscreen toolbar on every image; decorative art
   hides it (gallery keeps its buttons — figure download is useful).
4. `loop.svg` inlined into `gr.HTML` (file stays the source): avoids
   depending on Gradio's local-file-serving URL scheme; same ~3 KB.
5. `gr.Blocks(css=…)` emits a Gradio-6 `UserWarning` (params moved to
   `launch()`) but STILL applies: `launch()` falls back to the
   constructor value (`_deprecated_css`). Verified in 6.31.0 source +
   live `/config` render. Harmless; plan approach kept.
6. Chart recolor is a 1-line edit (`#4e79a7`→`#E9A13B` gold LLM,
   `#59a14f`→`#0E9F8A` teal greedy). No doc references the old hexes.
7. Sim scoreboard Dataframe moved into the "Retired sim lineage"
   accordion (with `tracks_note`); headline + recolored bars + footage
   table + serving stay visible; old flywheel block replaced by the
   exact null callout + v2 strip (same facts, plan copy). `key_points`
   kept visible. No `findings.json` byte changed.

## Replay determinism (PASS — resolves TODO(B)#1)

Re-ran on Space HEAD `86ed7f8` (Tab 3 unchanged by the rebase).
Bench-side (`.venv` python, Space root):

```
replay determinism PASS: 22 steps, clamp edges -1/N+1 OK
purity PASS: all 0..21 stable across repeated calls
cumulative at 0: tool calls so far: 0 · finds so far: 0
cumulative at 21: tool calls so far: 11 · finds so far: 3
```

`get_step` purity + clamp edges (−1→0, N+1→N) verified.
`gr.Timer(0.6, active=False)` + `.tick` + `gr.Timer(active=…)` toggle
verified on Gradio 6.31.0 (`Timer: True`). Live `/config` confirms
Timer ×1, Slider 0..21 on `86ed7f8`.

## Validation checklist §F2 (full sign-off, A+B)

- [x] `python -m py_compile app.py video_replay.py` green (Space root,
  `86ed7f8`).
- [x] JSON validity: `findings.json` + all 4 `samples/*/meta.json`
  load; gallery lists 4 titles; `.get` fallback on missing `eval_log`
  (no `KeyError`).
- [x] Media: every binary <50,000 bytes + new-binary total 218,940 B
  ≤ 300 KB (assert snippet); `PIL.Image.verify()` on each shipped
  image; part 2 adds zero binaries.
- [x] `grep -n "@spaces"` → docstring-only (3 hits, module docstring);
  hardware `cpu-basic` (runtime API); `requirements.txt` byte-identical
  to baseline (2 deps).
- [x] Billing: token-path grep → docstring + code-comment hits only;
  `resolve_viewer_token` untouched; no Space secrets added (variables
  `{}` via API; no Settings change).
- [x] Authenticated `/config`: HTTP 200, 117,660 bytes, Gradio 6.31.0,
  exact 4 labels (RUNNING @ `86ed7f8`).
- [x] Links: 3 HF URLs 200 signed-out (dataset ×2, weights); bench repo
  404 signed-out / 200 authed (PRIVATE — expected); Space 401
  signed-out / 200 authed (private — expected); no
  collection/writeup links; `VIDI` appears twice, both plan-mandated
  "blocked" notices (empty-state copy + contract, not links). Live
  config also contains Gradio-default `bus.png` placeholders for
  initially-empty Images (Gradio-injected, not displayed links).
- [x] Numbers audit: every displayed numeral ↔ §B4 or R1–R3 above;
  `0.580` formatting check passes; LADI-count rule holds (resolves
  TODO(B)#2).
- [x] Replay determinism PASS (purity + clamp edges, re-ran on
  `86ed7f8`; resolves TODO(B)#1).
- [x] Screenshots: before/after local (A) + live interactive 1440/390
  (B, 6 files), all committed next to this report (resolves TODO(B)#3).
- [x] `README.md` + `UPDATE_WORKFLOW.md` updated (showcase §, asset
  budget, replay §, screenshot §); `samples/SAMPLE_FORMAT.md` replay
  appendix (B); `assets/PROVENANCE.md` complete.

## Spend

$0.00 for #41 (no Modal/GPU, no HF Inference calls, no deploys).
Ledger rows `41 showcase-A cpu 0.5 $0.00` + `41 showcase-B cpu 0.5 $0.00`;
branch `docs/reports/modal-spend.md` union TOTAL $2.78 (includes
parallel issue-1 GPU spend, not #41 spend). Full pytest: 244 passed,
7 skipped (`.venv` python, `issue-41` worktree).
