---
license: {{SPDX_OR_OTHER}}
license_name: {{HUMAN_LICENSE_NAME}}
base_model: {{BASE_HF_ID}}
base_model_revision: {{BASE_REV_FULL}}
tags:
- {{TAG_1}}
- {{TAG_2}}
---

<!-- MODEL CARD TEMPLATE (issue #17). Fill every {{PLACEHOLDER}};
     none may survive to the Hub. -->

# {{MODEL_TITLE}}

{{ONE_PARAGRAPH_SUMMARY: base + method + data + what changed.}}

- **Base:** `{{BASE_HF_ID}}` @ `{{BASE_REV_SHORT}}`
- **Method:** {{METHOD}} ({{LORA_OR_FULL}})
- **Data:** `{{HF_DATASET_ID}}` @ `{{DATA_REV_SHORT}}`
  ({{N_SFT}} SFT / {{N_DPO}} DPO / {{N_RLVR}} RLVR)
- **Code:** repo commit `{{CODE_COMMIT}}`, train entry
  `{{TRAIN_ENTRY}}`

## Training config

| Param | Value |
|---|---|
| LoRA r / alpha / dropout | {{LORA_R}} / {{LORA_ALPHA}} / {{LORA_DROPOUT}} |
| Target modules | {{TARGET_MODULES}} |
| Epochs / batch / grad-accum | {{EPOCHS}} / {{BATCH}} / {{GRAD_ACCUM}} |
| LR / scheduler / warmup | {{LR}} / {{SCHED}} / {{WARMUP}} |
| Precision / max_len / seed | {{PRECISION}} / {{MAX_LEN}} / {{SEED}} |
| Hardware / steps / loss | {{HW}} / {{STEPS}} / {{LOSS}} |

Exact train commands (from the repo root):

```sh
{{TRAIN_COMMANDS}}
```

## Held-out results

`evals/flywheel_test.py`, seeds 101..110, {{BENCH_CONDITIONS}}:

| Arm | Mean | Stdev | n |
|---|---|---|---|
| Base | {{BASE_MEAN}} | {{BASE_STDEV}} | {{N}} |
| Finetuned | {{FT_MEAN}} | {{FT_STDEV}} | {{N}} |
| **Delta** | **{{DELTA}}** | — | — |

{{DELTA_VERDICT: null/significant, mechanism note.}}

## Safety gates

| Gate | Eval | Bar | Result |
|---|---|---|---|
| 1 — refusal | HarmBench + AIR-Bench | No drop vs base | {{G1}} |
| 2 — misuse | AgentHarm + redteam/safety | Harm down, benign flat | {{G2}} |
| 3 — embodied | SafeAgentBench + abstention | Pass + calibrated | {{G3}} |

## Serving

```sh
{{SERVE_COMMANDS}}
```

{{SERVING_NOTES: tokenizer pin, eager/timeout, parser.}}

## HF revision pin

- Model: `{{HF_MODEL_ID}}`
- Pinned revision: `{{HF_REV_FULL}}`
