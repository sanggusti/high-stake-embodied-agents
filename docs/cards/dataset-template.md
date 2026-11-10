---
license: {{SPDX_OR_OTHER}}
license_name: {{HUMAN_LICENSE_NAME}}
task_categories:
- {{TASK_1}}
- {{TASK_2}}
language:
- en
tags:
- {{TAG_1}}
- {{TAG_2}}
---

<!-- DATASET CARD TEMPLATE (issue #17). Fill every {{PLACEHOLDER}};
     none may survive to the Hub. See
     dmb-flywheel-v2-dataset.md for a filled example. -->

# {{DATASET_TITLE}}

{{ONE_PARAGRAPH_SUMMARY: what, for which loop, which split.}}

- **SFT** `sft.jsonl`: {{N_SFT}} chat samples ({{SFT_MIX}}).
- **DPO** `dpo.jsonl`: {{N_DPO}} pairs ({{DPO_MIX}}).
- **RLVR** `rlvr.jsonl`: {{N_RLVR}} episode reward vectors
  ({{N_SUCCESS}} success / {{N_FAILURE}} failure negatives).
- **Manifest** `manifest.json`: code commit, revs, per-source
  counts/splits/filters, SHA-256.

## Sources (provenance + license per source)

| Source | Upstream (pinned rev) | Split | Rows kept | License / terms |
|---|---|---|---|---|
| {{SRC}} | `{{HF_ID}}` @ `{{REV_SHORT}}` | {{SPLIT}} | {{KEPT}} | {{LICENSE}} ({{CLEARANCE_NOTE}}) |

Clearance: `hsea/data/clearance.py` (#24 terms); the exporter aborts
on any non-cleared source. Overall dataset: **{{OVERALL_TERMS}}**
{{GATING_NOTE}}.

## Repro

Exact export command (from the repo root):

```sh
{{EXPORT_COMMAND}}
```

Blind re-verification (no network, no writes):

```sh
{{VERIFY_COMMAND}}
```

## File hashes

| File | Rows | SHA-256 |
|---|---|---|
| `sft.jsonl` | {{N_SFT}} | `{{SHA_SFT}}` |
| `dpo.jsonl` | {{N_DPO}} | `{{SHA_DPO}}` |
| `rlvr.jsonl` | {{N_RLVR}} | `{{SHA_RLVR}}` |

## HF revision pin

- Dataset: `{{HF_DATASET_ID}}`{{GATED_FLAG}}
- Pinned revision: `{{HF_REV_FULL}}`
