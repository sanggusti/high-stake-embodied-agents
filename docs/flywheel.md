# Data Flywheel (Phase D)

The bench is also a data generator: teacher episodes → SFT/RLVR datasets →
finetune → held-out re-eval. This doc records the v1 loop end to end.

## Loop

1. **Teacher run** — `evals/flywheel_train.py` (forest_fire + tsunami ×
   train seeds 1..20 = 40 episodes) on the Modal teacher with
   `DMB_VISION=text`. Every episode records L2 (`dmb/watch/`) + L1 (.eval).
2. **Export** — `python -m dmb.data.export --log … --l2 … --out
   data/flywheel-v1 --split train --min-composite 0.5`:
   `sft.jsonl` (one chat sample per assistant tool-call turn, CoT kept),
   `rlvr.jsonl` (per-episode reward vectors), `manifest.json` (teacher,
   commit, splits, hashes). Split mixing is refused.
3. **Finetune** — `infra/modal/finetune.py` (TRL 1.16 LoRA r16, 3 epochs,
   A10G) → merged weights on volume `dmb-flywheel-models`.
4. **Re-eval** — `evals/flywheel_test.py` (held-out seeds 101..110, disjoint
   by construction) on base vs finetuned, same bench conditions.
5. **Publish** — dataset + weights on HuggingFace (links below).

## v1 results (2026-10-28)

- Teacher: Qwen2.5-7B-Instruct, 40 train episodes, mean composite 0.587,
  30/40 ≥ 0.5 → **224 SFT pairs** (LoRA r16, 3 epochs, 84 steps).
- Base held-out: mean **0.580** (stdev 0.158, n=20).
- Finetuned held-out: mean **0.531** (n=20, same seeds).
- **Delta: −0.049 ≈ −1 SE — statistically null.** The v1 flywheel is
  falsified at this data scale, exactly the failure mode frontier.md
  budgets for ("null/negative delta falsifies the flywheel, not the
  bench").

### Diagnosed mechanism: scan-spam + early submit

On 6/20 held-out seeds the finetuned model scores exactly 0.300: it spams
`request_edge_report` from the pad (burning snapshot budget without ever
arming) and submits. Single-teacher behavior cloning on 224 pairs
over-weighted the highest-frequency teacher action (scanning) and lost
the arm→takeoff→goto backbone on those seeds. Two seeds improved
strongly (+0.21/+0.37/+0.46), so the policy isn't uniformly damaged —
it's brittle. Prescription (flywheel v2, separate issue): 4-10× more
pairs across all six scenarios, plus DPO/RLVR negatives from the already
exported `rlvr.jsonl` failure episodes instead of pure success-filtered BC.

## Design notes

- **Text-policy v1.** The LM navigates on edge REPORTS (text), per the
  edge-first architecture; SFT teaches the action policy (reasoning +
  tool calls) with the vision path out of scope. Multimodal SFT (frame
  capture in L2 + TRL vision training) is the follow-up.
- **Grid bounds in brief.** The teacher initially flew to (50,70)-style
  coordinates off the 10×10 grid; stating bounds + start in the brief
  (map knowledge, not victim info) fixed it (0.31→0.83 on seed 1).
- **VL teacher attempt.** Qwen2.5-VL-7B-Instruct serves on A10G (16k ctx,
  mm-limit 8) but does not emit Hermes tool calls with vLLM 0.21 — it
  outputs pseudo-calls (`arm_drone_motors()`) as text. Forced tool_choice
  works, so the parser is fine; the weights don't volunteer the format.
  Retried text teacher instead (proven 19-call episodes).

## Artifacts

- Dataset: https://huggingface.co/datasets/sanggusti/dmb-flywheel-v1
  (also vendored at `data/flywheel-v1/`, 1.3 MB)
- Weights: https://huggingface.co/sanggusti/dmb-qwen2.5-7b-flywheel-v1
  (merged bf16; tokenizer files pinned to base to dodge the v5 schema drift below)
- L2 trajectories: `runs/flytrain-v1`, `runs/flytest-base`, `runs/flytest-ft`
  (local only, gitignored; replay with `python -m dmb.watch.replay`)

## Serving gotchas (all hit live)

- Modal factory-pattern apps don't deploy (LocalFunctionError — serve()
  must be module-level); teacher and finetune servers are twin files.
- transformers 5.x (finetune env) writes a new tokenizer schema
  (`extra_special_tokens` as list, no inline chat template) that vLLM
  0.21's bundled transformers v4 cannot parse → `AttributeError: 'list'
  object has no attribute 'keys'` at boot. Fix: pin the merged dir's
  tokenizer files to the base model's (identical tokenizer).
- 14 GB single-file safetensors streams off the Modal volume in ~100 s;
  with CUDA graphs the boot exceeded the 10-min startup timeout, so the
  finetune server uses `--enforce-eager` + 20-min timeout.
