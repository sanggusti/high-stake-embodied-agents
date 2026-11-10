# Issue #25 — partial write-up (DRAFT ONLY, no PR)

Branch: `issue-25` (fresh `/tmp` worktree from `origin/main`; main
checkout untouched). Date: 2026-11-10. Spend: $0.00 (this issue).

## What

Paper-style draft `docs/research-writeup.md` covering **concluded
work only** with REAL numbers verified cell-for-cell against committed
reports/docs. Serving / training / matrix sections are TODO-pending
shells (§8) with exact columns to fill. HF collections are planned,
not created (§9). No PR opened, nothing merged (DRAFT ONLY).

## Coverage per #25 checkbox

| Checkbox (issue #25) | Status | Draft-complete part | Pending part + what it waits on |
|---|---|---|---|
| Write-up (method, datasets, evals, safety results, honest nulls) in `docs/` or HF Article | **DRAFT-partial** | §§1–7 + §12: method (DORA/playbook/gates/scorers), datasets (26+21, #24 clearance + 7 skips, DORA no-go), footage 2-VLM table, safety/abstention/red-team results, flywheel v1 null + v2 curation, 9 honest nulls — all numbers traced | Final pass waits on §§8–10: serving rows (#1–#5), training rows (#8–#11), matrix (#12), cards (#17); HF Article conversion is out of draft scope |
| Public HF model collection + dataset collection (links verified signed-out) | **PENDING — plan only** | §9: proposed names (`sanggusti/hsea-models`, `sanggusti/hsea-datasets`), member lists, known revs (v2 `b44237e1…`), TBD-live pins, signed-out verification plan | Creation waits on ≥1 training model landing (#8 and/or #9); must not publish an all-TBD model collection. No collections created on this branch (per draft scope) |
| Live demo (HF Space) + backup recording | **PARTIAL** | §10: Space `sanggusti/high-stakes-embodied-agents` private + verified at `050f93d` (#16: 2 real samples, findings.json, viewer-billed, cpu-basic) | Public flip + backup recording: neither exists yet. Waits on the demo owner to flip the Space public and record/attach the backup; then link both in the final write-up |
| Results tables cross-checked against eval logs | **DRAFT-partial** | Every concluded table verified vs its committed source (§12 appendix: 10+ eval ids + pins). Note: `.eval` logs / `runs/` are gitignored — fresh worktree has 0/141 logs from 2026-11-09 — so committed reports citing eval ids are the verifiable evidence | Pending tables (§8 shells) are empty by design; they get cross-checked when #1–#5/#8–#12 land and post eval ids |

## Verification performed on this branch

- Each cited table diffed against its source file: footage 2-VLM (4
  rows) vs `docs/footage-evals.md`; safety 12-row matrix vs
  `issue-14` report; abstention eval/grid/reliability vs `issue-21`
  report; red-team family/severity/eval-id tables vs `issue-22`
  report; v2 files/per-source/DPO vs `issue-7` report + v2 manifest
  full SHAs; clearance condensed 11-row vs `license-clearance-24.md`;
  DORA probe/frozen plan vs `issue-23` report; Space harvest vs
  `issue-16` report; spend rows vs `modal-spend.md`.
- No invented numbers: TODO cells literally read `TODO`/`TODO-pending`
  with the unblocking issue named; no placeholder scores.
- `pytest`: full suite green (docs-only change) — see Cost/Artifacts.
- Isolation: all work in `/tmp/expert-spork-issue-25` on branch
  `issue-25`; no Modal CLI; no unsandboxed shells; no secrets printed.

## Cost

- Estimate (pre-run): $0 — docs-only draft, no model calls, no
  deploys. Pre-run ledger TOTAL $0.50 → $0.50 + $0 < $50 → GO.
- Actual: $0.00. Ledger row `25 partial-writeup cpu 0.2 $0.00`
  appended to `/tmp/modal-spend.log` and mirrored in
  `docs/reports/modal-spend.md`.
- Running TOTAL after this issue: **$0.50**.

## Artifacts

- `docs/research-writeup.md` (new; §§0–12 + references).
- This report: `docs/reports/issue-25-partial-writeup.md`.
- `docs/reports/modal-spend.md`: ledger union + this issue's $0.00 row.
- Pushed branch: `issue-25` (no PR, no merge).

## Shutdown

No Modal apps started (docs-only issue) — nothing to stop. No GPU spend.
