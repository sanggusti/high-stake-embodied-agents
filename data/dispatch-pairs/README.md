# Dispatch-priority DPO pairs (issue #20)

Crisis-to-action labels → `(situation, better/worse dispatch)` preference
pairs for `flywheel-v2`: TREC-IS criticality + HumAID/CrisisMMD humanitarian
categories. Situation text is derived from labels only — no raw post text —
per Twitter redistribution constraints (see `LICENSE_NOTES.md`).

- **Builder:** `hsea/data/dispatch_pairs.py` (`situation_from_labels`,
  `make_pair`, `build_pairs`, `verify_export`); manifest conventions mirror
  `hsea/data/export.py` (code commit, per-source counts, SHA-256).
- **Provenance:** `docs/datasets.md` §2 action picks (strategy b);
  `docs/alignment-research.md` playbook §4 (SFT→DPO stack, staged table).

## Export (exact command for #7 curation)

Inputs are local JSONL files, one record per line:
`{"id": "<record id>", "labels": {...}}` (schemas in the module docstring).

```sh
.venv/bin/python -m hsea.data.dispatch_pairs \
  --trec-is <trec-is records.jsonl> \
  --humaid <humaid records.jsonl> \
  --crisismmd <crisismmd records.jsonl> \
  --out data/dispatch-pairs --split train
```

Repro gates (no writes):

```sh
.venv/bin/python -m hsea.data.dispatch_pairs ... --dry-run      # parse + counts
.venv/bin/python -m hsea.data.dispatch_pairs --out data/dispatch-pairs \
  --split train --verify-only                                    # hashes + schema
```

## Files

| File | Content |
|---|---|
| `dpo.jsonl` | Sample pairs from synthetic labels (24 records, P1/P2/P3 mix) |
| `manifest.json` | Code commit, split, per-source counts, file/pair SHA-256 |
| `LICENSE_NOTES.md` | Per-source license/ToS; #24 gates v2 inclusion |

Full-corpus exports overwrite `dpo.jsonl`/`manifest.json` in place; the sample
here documents the schema and keeps `--verify-only` runnable blind.
