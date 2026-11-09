# Issue #23 — DORA fit evaluation (gated access probe + subset repro plan)

Branch: `issue-23`. Date: 2026-11-09.
Verdict: **principled NO-GO on the subset repro** (gated content returns
403 for our token; unblock requires the user's personal gated-repo
acceptance). Repro plan is frozen and ready to run post-access.

## Acceptance criteria

| Criterion | Outcome |
|---|---|
| Access obtained (or blocker documented with dates); subset (≥10 tasks) repro attempted | Blocker documented 2026-11-09: `gated:auto` form not accepted by token owner; gated fetches 403 (probe table below). Subset repro defined (10 tasks) but not runnable without data |
| Repro results posted (score + trajectory notes) or principled no-go with reasons | Principled no-go: valid token + 403 "not in the authorized list" on every gated file; gate needs personal identity + checkbox acceptance |
| Borrow-vs-diverge update proposed in `docs/alignment-research.md` if findings differ | No borrow/diverge change: findings confirm §2. Added access-status annotation (§2) with date, mechanism, outcome |
| License respected: gated NC data never redistributed; design reference only | No gated bytes downloaded (only public README/LICENSE + public Apache-2.0 GitHub code); nothing redistributed |

## Access probe results

Token: `HUGGINGFACE_TOKEN` from local `.env` (SET; value never printed
or committed). Only statuses, byte counts, and SHA-256 digests recorded.

| # | Request | Auth | Status | Bytes | SHA-256 | Meaning |
|---|---|---|---|---|---|---|
| 01 | `GET /api/datasets/Kingdrone-Junjue/DORA` | no | 200 | 51862 | `ec13a0f6…38353b` | card metadata is public |
| 02 | same | yes | 200 | 51862 | `ec13a0f6…38353b` | identical hash: token adds no metadata visibility |
| 03 | `GET …/tree/main?recursive=false` | yes | 200 | — | `4a88aef2…5ae73c37` | top-level listing public (7 entries) |
| 04 | `GET /api/whoami-v2` | yes | 200 | — | `316716b3…289e474` | token is valid (status only recorded) |
| 05 | `…/resolve/main/README.md` | yes | 200 | 5480 | `57ed7565…8f5a9be` | README exempt from gate (public) |
| 06 | `…/resolve/main/LICENSE.md` | yes | 200 | 23309 | `f4adeee7…c6593b8` | LICENSE exempt from gate (public) |
| 07 | `…/resolve/main/manifest.json` | yes | **403** | 172 | `fa5221b7…9cd51d` | **gated: not in authorized list** |
| 08 | `…/resolve/main/tasks/t1_perception_and_assessment.json` | yes | **403** | 172 | `fa5221b7…9cd51d` | **gated: not in authorized list** |

403 body (verbatim, 172 bytes):
`Access to dataset Kingdrone-Junjue/DORA is restricted and you are not in the authorized list. Visit https://huggingface.co/datasets/Kingdrone-Junjue/DORA to ask for access.`

## Dataset facts (public metadata only, 2026-11-09)

| Field | Value |
|---|---|
| Repo | `Kingdrone-Junjue/DORA` @ `76897d47cc2e2baa46b7657ad9c707e21ff707d6` |
| Gate | `gated:auto`, `private:false` |
| Size | 17,636,953,453 bytes (~17.6 GB) |
| Files | 791 (`images/` 744, `tasks/` 27, `checkpoints/` 16, root 4) |
| Task files | `tasks/t1_perception_and_assessment.json` … `t5_multi_modal_report_synthesis.json` + `manifest.json` |
| License tag | `other` → `dora-multi-license` (see LICENSE.md table below) |
| Traction | 1310 downloads, 8 likes |

## Gate requirements (from public `cardData` + README)

Request access at <https://huggingface.co/datasets/Kingdrone-Junjue/DORA>
(auto-granted after submitting the form — but the form needs the
account owner's personal details, so the agent cannot complete it):

| Field | Type |
|---|---|
| Name / Affiliation / Email / Country | text / text / text / country |
| "I will use DORA for non-commercial research and follow the licenses of its sources" | checkbox |
| "I will not use DORA for military or intelligence targeting, surveillance of individuals or communities, or discriminatory pricing, underwriting or redlining" | checkbox |

Note: name/affiliation/email are shared with the DORA authors for
dataset communications. No new personal signups were created (#23 NOTE).

## No-go statement + unblock steps

The subset repro cannot run until a token whose owner accepted the gate
is available. To unblock:

1. Open <https://huggingface.co/datasets/Kingdrone-Junjue/DORA> and submit
   the access form (fields above).
2. Confirm the same probes 07/08 return 200 with the accepted token.
3. Run the frozen repro plan below (≥10-task subset, frozen prompts).

## Frozen subset-repro plan (ready post-access)

Frozen prompt protocol (public Apache-2.0 code,
`dora/agent/prompts.py`, 115 lines, SHA-256
`64f30be20ff453b289d2c0d0a7b327107f3a7bb0a4969cb5bf40c45dc2a195f7`):
byte-identical template, whole prompt sent as one user message;
AP and IF modes share it (IF adds the gold tool-name sequence).

- Subset: first 2 tasks (id order) of each `tasks/tN_*.json`, T1–T5 =
  **10 tasks**, pinned to dataset revision `76897d47…` above.
- Data: `python scripts/prepare_data.py` (~18 GB incl. ~12.7 GB
  perception checkpoints; `--no-checkpoints` is scoring-only).
  Public README procedure; no Modal needed for download.
- Run (CPU + API providers only; no Modal deploys): DORA's
  `run_benchmark.py` takes `--tasks t1 t3`-style dimension filters and
  OpenAI-compatible model configs (`provider: openai` + `base_url`), so
  point it at Baseten `muse-spark-open-1026` or HF Inference
  `openai-api/hf/…` (issue #6). Exact per-task limit flag [UNVERIFIED —
  confirm via `run_benchmark.py --help` post-access].
- Score: `python scripts/evaluate.py --run <dir>` → DORA score
  (per-field typed accuracy averaged per task/dim/overall) + trajectory
  metrics (Tool-Any/In-Order/Exact-Match, ParamAcc, Efficiency);
  reference: Gemini-3.0-Flash 53.74% AVG vs 80.48% gold ceiling (v1.0).
- Evidence: hash `eval.json` (SHA-256) + trajectory-notes table; no
  `hsea.aggregate record` applies (that ingests inspect-ai `.eval` logs;
  DORA runs are scored by `evaluate.py`). No eval ran on this issue, so
  no registry entry was written — verified, not omitted.

## Borrow-vs-diverge

No change proposed: the probe confirms `docs/alignment-research.md` §2
(gated, ~17.5 GB, NC design-reference-only). Only a factual
access-status annotation (date + `gated:auto` mechanism + 403 outcome)
was added to §2; the borrow list (gold trajectories, typed eval,
protocol grounding) and diverge list (no safety axis, NC/gated) stand.

## License compliance

- Downloaded bytes: only gate-exempt public files (README.md,
  LICENSE.md) + public GitHub code. Zero gated bytes fetched, stored,
  or redistributed; probe bodies live in session-local `/tmp` only.
- Per-part licenses (public LICENSE.md, SHA-256 above): code
  Apache-2.0 (except vendored DINOv3 + NVIDIA SegFormer NC parts);
  `tasks/` annotations CC BY-NC-SA 4.0; `images/` per-source (xBD/Maxar,
  DisasterM3, OSM ODbL, Planet CC BY-SA, GVLM-CD NC-only, RescueNet NC).
  DORA stays a design reference + future eval comparison, never ingested.

## Verbatim commands

Actual probes run 2026-11-09 (token read from `.env` inside the
scripts, never echoed):

```bash
python /tmp/issue-23-probe.py /tmp/issue-23-probe       # probes 01-04: metadata, tree, whoami
python /tmp/issue-23-dlprobe.py /tmp/issue-23-probe     # probes 05-08: resolve README/LICENSE/manifest/t1
curl -sL -o dora-prompts.py https://raw.githubusercontent.com/Junjue-Wang/DORA/master/dora/agent/prompts.py
curl -sL -o dora-readme.md https://raw.githubusercontent.com/Junjue-Wang/DORA/master/README.md
curl -sL -o dora-license-full.md "https://huggingface.co/datasets/Kingdrone-Junjue/DORA/resolve/main/LICENSE.md"
sha256sum dora-prompts.py dora-readme.md dora-license-full.md
```

Curl equivalents of the gated checks (reproducible with any token):

```bash
export HF_TOKEN=$HUGGINGFACE_TOKEN   # names only; value stays in .env
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/datasets/Kingdrone-Junjue/DORA/resolve/main/manifest.json
# -> 403 before gate acceptance; expect 200 after
```

Probe scripts are session-local (`/tmp/issue-23-probe*.py`, not
committed); the statuses + digests above are the committed evidence.

## Cost

- Estimate (pre-run): $0 — Hub metadata/resolve API calls are free; no
  inference, no Modal. Pre-run ledger TOTAL $0.50, so $0.50 + $0 < $50 → GO.
- Actual: $0.00. Ledger row `23 dora-access-probe api 0.1 $0.00`.
- Running TOTAL after this issue: **$0.50**.

## Plots

None — and intentionally so. No repro runs executed (gated data
blocked all task access), so no scores exist to distribute or compare.
Per the issue NOTE, plots attach to repro runs (score distribution);
a status-count bar over 8 probes would be chartjunk, so it is omitted
in favor of the probe table above.

## Artifacts

- This report: `docs/reports/issue-23-dora-fit.md` (no `.png`: nothing plottable, see above).
- `docs/alignment-research.md` §2: access-status annotation.
- `docs/reports/modal-spend.md`: ledger union + this issue's $0.00 row.

## Shutdown

No Modal apps started (explicitly out of scope for this issue) —
nothing to stop. No GPU spend.
