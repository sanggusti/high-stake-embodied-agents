# License / ToS notes — dispatch-priority pairs (issue #20)

Research use only where noted. Only #24-cleared sources may ship in
flywheel-v2 (#7); this pipeline can run on local placeholders until then.

| Source | License / terms | Redistribution |
|---|---|---|
| HumAID (`QCRI/HumAID-all` mirror; primary gated by CrisisNLP) | CC-BY-NC-SA-4.0 | Research-only, non-commercial, share-alike |
| CrisisMMD (`QCRI/CrisisMMD` v2.0) | CC-BY-NC-SA-4.0 | Research-only, non-commercial, share-alike |
| TREC-IS (no official HF; NIST track data) | **SKIPPED per user decision 2026-11-09 (X API key + ToS)** — was UNVERIFIED pending #24 | Excluded from flywheel-v2 until revisited; HumAID/CrisisMMD pairs only |

Twitter ToS: tweet text is not redistributable. The builder is label-derived
only — it drops unknown record fields (including any `text`), and committed
exports carry situation templates + label provenance, never post content.
Mirror label fidelity (HumAID primary vs `QCRI/HumAID-all`) is unconfirmed;
unmapped categories fall back to routine (P2) review rather than failing.
