# Cards (issue #17)

Reusable templates plus per-release card content. Templates are the
contract for every future run; per-release files are the filled,
reviewed copy that gets uploaded to the Hub.

| File | What |
|---|---|
| [dataset-template.md](dataset-template.md) | Template for dataset cards (front-matter + sections) |
| [model-template.md](model-template.md) | Template for model cards (front-matter + sections) |
| [dmb-flywheel-v2-dataset.md](dmb-flywheel-v2-dataset.md) | Flywheel-v2 dataset card content (rev `b44237e`, manifest counts) |
| [dmb-flywheel-v2-model-TODO.md](dmb-flywheel-v2-model-TODO.md) | First-finetune model card — TODO stub (no finetune exists yet) |

Rules: fill every `{{PLACEHOLDER}}` (none may survive to the Hub);
quote the pinned HF revision and the exact export/train commands;
keep the license/clearance table per source; update the matching
`data/*/README.md` in the same PR.
