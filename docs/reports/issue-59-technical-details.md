# Issue #59 — Space Technical details tab + diagrams, drop This-Space self-link

Branch: `issue-59`. Date: 2026-11-10. Scope: add a 5th `Technical details`
tab to the HF Space (experiment design, tools, eval tools, datasets,
finetunings, alignments, each with explanation plus trace line), ship 3
original technical diagrams, and remove the `This Space` self-link from
the header links, per
[issue #59](https://github.com/sanggusti/expert-spork/issues/59).
Owner copy rules from the
[owner-feedback comment on #41](https://github.com/sanggusti/expert-spork/issues/41#issuecomment-6760338597)
are binding and hold (audit below). Bench vintage: `origin/main`
`7ef8744`. No Modal/GPU. No PR opened, no merge, #59 left open.

Sources read: `WORKFLOW.md`, issue #59 (full body), the #41 feedback
comment, `docs/process.md`, `docs/research-track.md`,
`docs/alignment-methods.md` plus the 3 lane notes
(`alignment-methods-posttraining/vlm/safety.md`), `docs/datasets.md`,
`docs/inference.md`, `docs/safety-gates.md`, `evals/` (5 tasks plus
smoke), `hsea/scorers/` (4 scorers), `hsea/aggregate.py`,
`hsea/data/curate_v2.py`, and `scripts/` (all CLIs verified against
code, no guessed flags).

## Space commits

- `2bb730f` (before, #57) → **`b3c0b26`** (this pass, pushed: tech tab
  plus diagrams plus self-link removal).
- Live state (`b3c0b26` RUNNING): `/config` HTTP 200, 164,385
  bytes, Gradio 6.31.0, tabs exactly `Overview` / `Findings` /
  `Interactive examples` / `Inference demo` / `Technical details`.
  Hardware `cpu-basic`.
- Diff shape (`2bb730f..b3c0b26`): 7 files changed, +200/−13 (`app.py`,
  `README.md`, `UPDATE_WORKFLOW.md`, `assets/PROVENANCE.md`, 3 new
  PNGs). Token path, replay logic, `requirements.txt`,
  `findings.json`, `theme.css`, samples untouched; Tab 1/3/4 blocks
  untouched (app.py hunks cover docstring, `BENCH_PIN`, links,
  honesty, spend vintage, tech constants, and the new tab only).

## Content map (new tab)

Intro plus 6 sections, each a short plain explanation plus a trace
line; 3 one-paragraph bridges; 3 figures each with prose above and
below plus a caption line; one takeaway with trace.

| # | Block | Prose | Visual | Trace |
|---|---|---|---|---|
| — | Intro | what the tab covers, figures flagged original | — | — (no numerals) |
| 1 | Experiment design | 5 stages, 3 gates with bars, fail returns upstream | — | research-track §4; process §3; safety-gates |
| F1 | Loop diagram | caption + bridge to command groups | fig-tech-loop.png | same as §1 |
| 2 | Tools | inspect eval, aggregate record/table/uplift, run_safety, curate_v2 | — | process §§1/5; aggregate.py; run_safety.py; curate_v2.py |
| 3 | Eval tools | 5 task/scorer pairs plus smoke routing-only | — | evals/; hsea/scorers/; #14/#21/#22 |
| F2 | Harness map | caption + bridge on score flow | fig-tech-harness.png | same as §3 |
| 4 | Datasets | footage 5, action 5, safety 2, one-line blocked notes | — | datasets §§1–2; research-track §2; clearance-24 |
| F3 | Data-flow diagram | caption + bridge on stage order | fig-tech-dataflow.png | #7 (split sizes) |
| 5 | Finetunings | SFT, E-DPO bake-off, LoRA pattern, ORPO/distill, E-RLVR | — | process §2; finetune.py; #7; posttraining note; lanes |
| 6 | Alignments | 3 lane surveys with counts and leads | — | alignment-methods.md index + 3 lane notes |
| — | Takeaway | five gated stages, deterministic scorers, ranked methods | — | research-track; process; alignment-methods |

## Diagram list + provenance

Original matplotlib diagrams (Emberline palette, DejaVu Sans, dpi
100), rendered bench-side by the committed
`scripts/plot_tech_details.py` (matplotlib 3.10.9, Python 3.10.4,
deterministic, no random data). No copied paper figures.

| Space file | Subject | Dims | Bytes | SHA-256 |
|---|---|---|---|---|
| `fig-tech-loop.png` | Lexsi loop pipeline with 3 safety gates | 900x340 | 34,728 | `76e910e81e7652be…` |
| `fig-tech-harness.png` | eval-harness component map (tasks → scorers → logs → tables/plots) | 900x380 | 47,147 | `bd85a242e6b13d6d…` |
| `fig-tech-dataflow.png` | curation → tune → eval data flow with v2 split sizes | 900x380 | 47,611 | `b7d1db3e9d7f5747…` |

New-binary total: 129,486 bytes (budget ≤300 KB); every file <50,000
bytes with `PIL.Image.verify()` green. Full recipe plus SHAs in Space
`assets/PROVENANCE.md` (issue-59 section). Layout QA: boxes, diamonds,
and arrows inspected pixel-side before push (label-overflow and arrow
landing fixes re-rendered in the committed script).

## Self-link removal

`LINKS_MD` in `app.py` drops the 4th link
(`[🚁 This Space](https://huggingface.co/spaces/…)`) and keeps Bench
repo plus both dataset links. `spaces/sanggusti/high-stakes-embodied-agents`
count: 0 in `app.py`, 0 in live `/config`; the 1 remaining repo hit is
the `UPDATE_WORKFLOW.md` §5 runtime-API curl example (dev doc, never
rendered). The hero prose `This Space tracks the loop` is not a link
and stays.

## Number-trace audit (every displayed numeral in the new tab)

Carry-over rows F1–F4, T1–T3, V1, V2, N10, N11 from
`docs/reports/issue-41-showcase.md` §B4 (via `issue-41-feedback.md` and
`issue-57-findings-narrative.md`). New numerals below, each with its
source:

| ID | Number as shown | Source |
|---|---|---|
| T59-G1 | 5 stages; 3 gates; 0.95 / 0.02 / 1.0 / 0.85 / 0.80 / 0.60 / 0.10 / 0.90 | research-track §4 (stages/gates); process §3 + safety-gates.md (bars verbatim) |
| T59-T1 | 78 items; 6 benches | #14 report (T1 reuse; `BENCHES` 6-tuple in `hsea/safety/data.py`) |
| T59-T2 | 60 risk + 20 benign | #21 report (T2 reuse; `docs/abstention-evals.md` set line) |
| T59-T3 | 30 attacks, 3 families | #22 report (T3 reuse; `data/redteam/attacks.jsonl` 30 lines) |
| T59-V2 | 8,673 / 7,000 / 250 (prose + data-flow figure) | #7 report (V2/N10 reuse; `data/flywheel-v2/manifest.json` totals) |
| T59-LoRA | r16, alpha 32, 3 epochs, A10G | process §2 + `infra/modal/finetune.py` (r16/a32/dropout 0.05/3ep/b2x4ac/lr 2e-4; shown subset quoted) |
| T59-ALN | 27 methods / 9 families; 21 / 7; 10 methods | alignment-methods.md lane summaries (posttraining §1–9, VLM §1–7, safety §1–10) |
| T59-VIN | bench `7ef8744`; TOTAL $6.29 | origin/main HEAD at build; modal-spend.md union ($6.29 incl. this $0.00 row) |

Discipline holds: no LADI counts (21/24, 22/24), no Qwen 3/10, no S1
numbers, no spend math beyond the kept Findings rows; LoRA subset (r16,
alpha 32, 3 epochs) is quoted from the recipe without inventing the
rest; figure numerals (8,673 / 7,000 / 250) re-trace to #7.

## Copy-rule audit (Space tree + live `/config`)

- Em-dash: 0 (literal `—` and `u2014` greps over `app.py`,
  `video_replay.py`, `theme.css`, `README.md`, `UPDATE_WORKFLOW.md`,
  samples, PROVENANCE, loop.svg); live `/config` count 0. New TECH
  block additionally free of `but`, `→`, and `–`.
- AI-tell phrasing: 0 (`not X but`, `not just`, `more than just`,
  `in today`, `delve`, `landscape` greps). Plain `never` without `but`
  remains (e.g. `never counts as a result`).
- Title: HSEA wordmark present; `expert-spork` 2 source hits, both the
  kept repo URL (renders 3×: header, hero, overview bottom); live
  `/config` 3 hits, all the repo URL. Zero in prose.
- Models: `Qwen2.5` / `Mistral-7B` 0 in user-visible strings and live
  config; `dmb-qwen2.5-7b-flywheel-v1` 0. Shown names are lanes plus
  run provenance only. The 3 `findings.json` frozen hits stay
  unrendered (verified: only a code comment references those fields).
- Tracks: dataset-composed names only; `forest_fire` in the allowed
  retired note plus 2 pre-existing code comments; `VIDI` twice as
  blocked-note text with no URL; no weights/collection/writeup links.
- Untouched: token-path grep is docstring plus code-comment hits only;
  `@spaces` docstring only; `requirements.txt` (`authlib` +
  `itsdangerous`), `findings.json`, `theme.css`, replay code
  byte-identical; first 4 tab labels frozen.

## Links (signed-out curl)

v2 dataset 200, v1 dataset 200, bench repo 404 (PRIVATE, expected),
Space `/config` 404 signed-out with authed `/config` 200 (private,
expected). No weights, collection, or VIDI links.

## Screenshots (live Space `b3c0b26`, authenticated, Gradio 6.31.0, `?__theme=light`)

| File | Viewport | Tab | Notes |
|---|---|---|---|
| `issue-59-technical-1440.png` | 1440px | Technical details (full page) | 6 sections, 3 interleaved captioned figures, bridges, takeaway, links without This Space |
| `issue-59-technical-390.png` | 390px | Technical details (full page) | Text wraps, figures scale, no page overflow; tab reached via More-tabs overflow |

![technical 1440](issue-59-technical-1440.png)
![technical narrow](issue-59-technical-390.png)

Render notes: hero legible; header shows Bench repo plus v2/v1 only;
vintage reads bench `7ef8744`; all 3 figures legible with captions
below; no adjacent visuals (asserted programmatically, see below).
Narrow tab-bar collapse into the `More tabs` overflow (2 visible tabs)
is the pre-existing Gradio behavior noted in #41, not a regression;
the active tab promotes into the visible strip. Tooling: `playwright`
in `/tmp/shot-venv` plus checked-in Chromium
(`PLAYWRIGHT_BROWSERS_PATH=/tmp/ms-playwright`); Chromium launch needed
one-time unsandboxed approval; live auth via `Authorization: Bearer`
header (token from bench `.env`, never printed). Tab click needed a
forced click (overlay interception in headless); narrow needed the
overflow menu path.

## Read-through verdict

Coherent: each of the 6 sections states what the thing is, names the
exact commands or files, and ends in a trace line; figures interleave
with prose (design → loop → tools → eval → harness → datasets →
data-flow → finetunes → alignments → takeaway); every numeral traces to
the table above; the takeaway restates the loop in one sentence. No
stacking, no filler sections, no invented CLIs (all flags verified
against `hsea/`, `evals/`, `scripts/`).

## Verification log

- [x] `python -m py_compile app.py video_replay.py harvest_sample.py` green (Space root, `b3c0b26`).
- [x] JSON validity: `findings.json` + the 1 remaining `samples/*/meta.json` load.
- [x] Media: every binary <50,000 bytes (11 files incl. 3 new); new drop 129,486 B (budget 300 KB); `PIL.Image.verify()` on each image.
- [x] `grep -n "@spaces"` docstring-only (3 hits); hardware `cpu-basic`; `requirements.txt` unchanged.
- [x] Billing: token-path grep docstring plus comment only; no Space secrets added.
- [x] Structure check (real `build_demo` via recording stub): 5 exact tab labels, 3 images + 0 galleries + 0 dataframes in the new tab, no two visuals adjacent, prose on both sides of every visual, 6 headings + 1 takeaway + 7 traces, image files exist. PASS.
- [x] Authenticated `/config`: HTTP 200, 164,385 bytes, Gradio 6.31.0, exact 5 labels; 8 takeaways + 20 traces + `TOTAL $6.29` + `bench 7ef8744` present; galleries 0, images 11.
- [x] Links: datasets 200 signed-out; bench repo 404 and Space 404 signed-out (both private, expected).
- [x] Numbers audit: every displayed numeral traces to the table above; LADI-count rule holds.
- [x] Copy-rule greps: em-dash 0, AI-tells 0, legacy models 0, repo-URL-only `expert-spork`, self-link 0 in UI/config, `forest_fire` in the retired note only.
- [x] Screenshots: desktop 1440 + narrow 390 committed next to this report; read-through verdict above.
- [x] Full pytest: 244 passed, 7 skipped (main-repo `.venv` python, this worktree).

## Spend

$0.00 for this task (no Modal/GPU, no HF Inference calls, no
deploys). Ledger row `59 tech-details cpu 0.5 $0.00` appended to
`/tmp/modal-spend.log` and mirrored by union; branch
`docs/reports/modal-spend.md` TOTAL $6.29 (includes parallel serving
GPU spend on issues 1/2/3, not task spend). The Space spend row pins
the same union at bench `7ef8744` (TOTAL $6.29).
