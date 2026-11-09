# DMB Flywheel v1 — teacher trajectories (train split)

Post-training data from Disaster Mitigation Bench (Phase D): success-filtered
drone-navigator episodes on train seeds, exported from inspect-ai eval logs
(L1 transcripts + scores) with L2 trajectory references.

- **Teacher:** Qwen/Qwen2.5-7B-Instruct (Modal vLLM 0.21.0, A10G, Hermes tools)
- **Bench conditions:** `DMB_VISION=text` (text-policy: LM navigates on edge
  reports), mock backend, grid bounds in brief
- **Scenarios:** forest_fire + tsunami × train seeds 1..20 (40 episodes)
- **Filter:** composite ≥ 0.5 → 30 episodes, 224 SFT pairs
- **Held-out:** test seeds 101..110 (disjoint by construction)

## Files

| File | Content |
|---|---|
| `sft.jsonl` | One chat sample per assistant tool-call turn (prefix + turn, CoT kept) |
| `rlvr.jsonl` | One record per episode: reward vector, seed, world hash, L2 ref |
| `manifest.json` | Teacher, code commit, splits, filters, file SHA-256, world hashes |

## Determinism

`manifest.json` pins the generating code commit, the eval log, per-sample
world hashes, and file hashes. Re-running the same seeds on the same commit
reproduces the same worlds; teacher sampling is the only nondeterminism.

Source: Disaster Mitigation Bench, `dmb/data/export.py`.
