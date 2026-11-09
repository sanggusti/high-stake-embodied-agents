# Issue-PR-Merge Workflow

No direct pushes to `main`. Every change flows: **Issue -> Branch -> PR -> Merge -> Close**.

## 1. Branches

- `main` is protected; never push to it directly.
- Naming: `<type>/<issue#>-<slug>` (e.g. `feat/12-mcp-ras-area`, `fix/34-point-scorer-gsd`, `docs/07-readme-pipeline`, `chore/09-modal-volumes`).
- One issue per branch. Rebase on `main` before opening the PR.

## 2. Issues

- Title + Summary + Acceptance criteria (including the scorer/metric and the repro gate that proves it) + Provenance (donor path or DORA/Lexsi reference) + Test plan.
- Labels: type (`feat`/`fix`/`docs`/`chore`), area (`bench`/`training`/`eval`/`serving`/`datasets`), priority.
- Seeding follow-ups link SEEDING.md manifest lines (COPY/ADAPT/SKIP path + refs).

## 3. Commits

- Conventional Commits: `feat`/`fix`/`docs`/`test`/`chore`/`refactor` + scope + issue ref, e.g. `feat(scorers): add polygon IoU tau 0.5 (#12)`.
- Small atomic commits. No secrets (`.env` is never committed; keys via environment only).

## 4. PRs

- Template (`.github/pull_request_template.md`): What / Why / Provenance / How-tested (inspect eval id + fingerprints + trajectory table) / Before-after metrics (>= 2 metrics) / Risk + rollback.
- Must link its closing issue (`Closes #N`). Keep to <= 400 LOC excluding generated files.
- CI must be green: lint + typecheck, touched-area tests, `--sanity` gold-vs-gold check, manifest SHA-256 verify, no-gated-data leak check. At least one review. No merge on red.

## 5. Merge

- Squash-merge to `main` with the PR title as message. Delete the branch after merge.
- Tag data/model releases (`data-vX.Y`, `model-vX.Y`) with the HF revision pin; update manifests + dataset/model cards in the same PR.

## 6. Close

- Issues close only via merged PR (auto-close keyword) after CI green + reviewer approval.
- Verify acceptance criteria including repro gates (`prepare_data --verify-only`, `--dry-run`, `--sanity`) and updated HF collections.
- Post demo evidence (live demo link + ROI table) on the milestone issue. Reopen (never file a dupe) on regression, with the failing fingerprint attached.
