# HSEA on Hugging Face — master index (issue #75)

All HSEA artifacts are discoverable under
[`sanggusti`](https://huggingface.co/sanggusti) on the Hub. Revision pins
below are full commit SHAs; re-pin after every re-upload.

## 1. Datasets

| Repo | Contents | Pin |
|---|---|---|
| [`sanggusti/dmb-flywheel-v1`](https://huggingface.co/datasets/sanggusti/dmb-flywheel-v1) | Flywheel v1 SFT + RLVR (train) | `fc1d917486c8553dc4b64d40306ab95cf286ed6d` |
| [`sanggusti/dmb-flywheel-v2`](https://huggingface.co/datasets/sanggusti/dmb-flywheel-v2) | Flywheel v2 SFT (8,673) + DPO (7,000) + RLVR (250) | `9ea14ba63cda2b1f180ff4858a51ba99590f1ebf` |
| [`sanggusti/hsea-datasets`](https://huggingface.co/datasets/sanggusti/hsea-datasets) | `abstention/` (60+20, #21) · `dispatch-pairs/` (24 DPO, #20) · `redteam/` (30 attacks, #22) · `firm/` (NASA FIRMS zip) | `cc669f29f50b53c3655769a966fc665f3dbc128d` |
| [`sanggusti/hsea-logs`](https://huggingface.co/datasets/sanggusti/hsea-logs) | `runs/` (123 run dirs: config+scores) · `logs/` (141 `.eval` archives) · `reports/` (26 write-ups + 64 plots) | `6da307b2ea4fd21f11e6aced6f73cfbabb91fc77` |

Training agents compose `dmb-flywheel-v2` **plus** the `hsea-datasets`
safety/grounding subsets (#73, #74). flywheel-v1/v2 data files are
byte-identical to the repo `data/` copies (SHA-256 verified at upload;
only the v2 card pin line was refreshed).

## 2. Models / adapters

| Repo | Contents | Owner | Pin |
|---|---|---|---|
| [`sanggusti/dmb-qwen2.5-7b-flywheel-v1`](https://huggingface.co/sanggusti/dmb-qwen2.5-7b-flywheel-v1) | Flywheel-v1 finetune (pre-existing) | flywheel-v1 training | `1d89f6756d38e8fd4be9e08c9fa82841662eb976` |
| Qwen3.8-27B AlignTune adapter | SFT on flywheel-v2 + safety gates | #8 (uploads its own) | _pending — backfill when #8 lands_ |
| Tiny Aya Base LoRA + alignment | SFT + SafeTune/DPO loop | #73 (uploads its own) | _pending — backfill when #73 lands_ |
| Aya Expanse 8B LoRA + alignment | SFT + SafeTune/DPO loop | #74 (uploads its own) | _pending — backfill when #74 lands_ |

This issue never duplicate-uploads adapters owned by #8/#73/#74; this
table is backfilled with their repo URLs + pins once they land.

## 3. Model-card checklist (for training-agent uploads)

Every adapter repo must ship a `README.md` card with: base model +
revision, training-data repos + revisions (`dmb-flywheel-v2` rev,
`safety` subset revs), method (LoRA rank/targets, SFT→DPO/SafeTune
stages), eval summary (held-out split, ≥2 metrics, HarmBench/AIR-Bench
refusal vs base, eval ids), license + research-only note, and a pointer
back to this index.

## 4. Reproduce / download

```sh
hf download sanggusti/dmb-flywheel-v2 --repo-type dataset --local-dir dmb-flywheel-v2
hf download sanggusti/hsea-datasets --repo-type dataset --local-dir hsea-datasets
hf download sanggusti/hsea-logs --repo-type dataset --local-dir hsea-logs
```

## 5. Upload log

- 2026-11-10 (#75): created `hsea-datasets` + `hsea-logs`; uploaded
  safety/grounding sets, runs/logs/reports mirror, dataset cards; synced
  `dmb-flywheel-v2` card pin line. No GPU spend.
