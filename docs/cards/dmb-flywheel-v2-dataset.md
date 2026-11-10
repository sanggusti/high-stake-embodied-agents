---
license: other
license_name: mixed-per-source-research-only
task_categories:
- text-generation
- visual-question-answering
language:
- en
tags:
- disaster-response
- sft
- dpo
- rlvr
---

<!-- DATASET CARD CONTENT for sanggusti/dmb-flywheel-v2 (issue #17).
     Counts + hashes from data/flywheel-v2/manifest.json; rev pin
     b44237e. Upload to the Hub is REMAINING work (see report
     docs/reports/issue-17-partial-runbook.md). -->

# DMB Flywheel v2 — curated post-training data (train split)

Multi-source SFT + dispatch-DPO + RLVR curation for the High-stakes
Embodied Agents (HSEA) flywheel, issue #7. All upstream rows come from
pinned revisions (train side only — no overlap with the #19 footage
eval splits); pixels and post text stay upstream (refs/labels only).

- **SFT** `sft.jsonl`: 8,673 chat samples (footage QA, DisasterM3
  instructions, CrisiText notify-head, bench tool-call turns).
- **DPO** `dpo.jsonl`: 7,000 dispatch-priority pairs (HumAID 4,000 +
  CrisisMMD 3,000, labels only; TREC-IS skipped per #24).
- **RLVR** `rlvr.jsonl`: 250 episode reward vectors (133 success /
  117 failure negatives), incl. the v1 failure episodes.
- **Manifest** `manifest.json`: code commit, revs, per-source
  counts/splits/filters, SHA-256.

## Sources (provenance + license per source)

| Source | Upstream (pinned rev) | Split | Rows kept | License / terms |
|---|---|---|---|---|
| floodnet-vqa | `ljx620/FloodNet-VQA` @ `ec9277af` | train | 1,000 QA | CDLA-Permissive-1.0 (base FloodNet; mirror card untagged — QA text + image refs only, gated research-only) |
| ladi-v2 | `MITLL/LADI-v2-dataset` @ `5f2dbfe8` | train | 2,400 QA (200 imgs × 12) | CC-BY-4.0 (official MITLL upload) |
| disasterm3 | `Kingdrone-Junjue/DisasterM3` @ `1da6da73` (`DisasterM3_Instruct/train_release.json`) | train file | 3,000 (375 × 8 tasks) | CC-BY-NC-SA-4.0 (research-only, non-commercial, share-alike) |
| crisitext | `LanD-FBK/crisitext` @ `33c5545c` | train | 2,022 msgs / 300 situations | CC-BY-4.0 (FEMA/GTD rows only; synthetic `generated` rows excluded; guideline text joined via pinned `fema_guidelines.json`) |
| humaid | `QCRI/HumAID-all` @ `93470f41` | train | 4,000 DPO (P1 915 / P2 2,126 / P3 959) | CC-BY-NC-SA-4.0 (research-only; labels-only, no tweet text) |
| crisismmd | `QCRI/CrisisMMD` @ `10a5626b` | train | 3,000 DPO (of 8,409 joined; P1 313 / P2 1,209 / P3 1,478) | CC-BY-NC-SA-4.0 (research-only; humanitarian+informative joined by tweet_id, labels-only) |
| l2 | `logs/*.eval` (157 logs, 210 eps) + `data/flywheel-v1` (224 SFT / 40 RLVR) | train seeds | 251 SFT (27 fresh + 224 v1) / 250 RLVR | own bench data (SFT: composite ≥ 0.5, non-mock teachers; RLVR: all scored incl. failures) |

Clearance: `hsea/data/clearance.py` (#24 terms); the exporter aborts
on any non-cleared source. Overall dataset: **research-only** (NC/SA
components) — gated access, see repo `docs/license-clearance-24.md`.

## Repro

Exact export command (from the repo root):

```sh
.venv/bin/python -m hsea.data.curate_v2 --out data/flywheel-v2 --split train
```

Blind re-verification (no network, no writes):

```sh
.venv/bin/python -m hsea.data.curate_v2 --out data/flywheel-v2 \
  --split train --verify-only
```

## File hashes

| File | Rows | SHA-256 |
|---|---|---|
| `sft.jsonl` | 8,673 | `5b944648ad57ef0483988aa2811d7b0db881e2e8aa81352313f486daf91e6424` |
| `dpo.jsonl` | 7,000 | `a322461981f872989522d76bd7cfde971bc12d5561ed255b7d2837d92d136f59` |
| `rlvr.jsonl` | 250 | `7a9f6ad838e16e8035be06e1ee9202702c28381059bd868d673baab23d215096` |

## HF revision pin

- Dataset: `sanggusti/dmb-flywheel-v2` (gated)
- Pinned revision: `b44237e145af1f0d9cfa3f22ec36f77d22ca5393`
