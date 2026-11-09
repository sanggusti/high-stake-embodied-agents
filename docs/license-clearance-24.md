# Dataset License Clearance Sweep — Issue #24 (2026-11-09)

Resolves every `[UNVERIFIED]` license in the `docs/datasets.md` shortlist
from primary sources. Nothing trains without clearance.
Clearance table lives here and is back-ported to `docs/datasets.md`.

Disposition key: **CLEARED** = citable by #7/#8 for finetuning (within any
listed constraint) · **BLOCKED** = must not train · **NEEDS-LEGAL** = agent
cannot decide, needs counsel/user action.

## 1. Clearance table

| # | Dataset | License (primary source) | Disposition | Primary-source URL |
|---|---|---|---|---|
| 1 | VIDI | none stated (full README has zero license mentions; GitHub license API 404; paper arXiv:2205.13277 states no dataset license) | BLOCKED / NEEDS-LEGAL — YouTube ToS + original uploaders hold rights, no redistribution grant | https://raw.githubusercontent.com/vididataset/VIDI/main/README.md |
| 2 | Sen1Floods11 | none stated (full README has zero license mentions; GitHub license API 404; open GCS bucket; underlying Sentinel-1/2 is Copernicus free-and-open, 446 hand labels unlicensed) | BLOCKED / NEEDS-LEGAL until confirmed in writing — SKIPPED (user decision 2026-11-09; needs author email) | https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/README.md |
| 3 | DAD (ASONAM17 damage images) | CrisisNLP Terms of Use: research on humanitarian computing only; same tarball bundled in MEDIC under CC BY-NC-SA 4.0 | Research-only terms — SKIPPED (user decision 2026-11-09; terms capture outstanding); use MEDIC bundle instead | https://crisisnlp.qcri.org/data/ASONAM17_damage_images/ASNOM17_Damage_Image_Dataset.tar.gz |
| 4 | DMD (Mouzannar et al. 2018) | CC BY 4.0 ("licensed under CC BY 4.0", DOI 10.24432/C52P6P, direct 1.1 GB download) | CLEARED (attribute Mouzannar/Rizk/Awad 2018) | https://archive.ics.uci.edu/dataset/456/disaster+management+through+sentiment+analysis+using+deep+learning+and+big+data |
| 5 | SARD | CC BY 4.0 (DataPort License field = Creative Commons Attribution, DOI 10.21227/ahxm-k331) but **Standard Dataset** = paid subscription to access files | SKIPPED (user decision 2026-11-09; paid signup) — was: cleared license-wise, gated | https://ieee-dataport.org/documents/search?search=SARD (dataset DOI 10.21227/ahxm-k331) |
| 6 | HERIDAL | CC Attribution 3.0 Unported ("freely available to browse, download, and use for commercial, scientific and educational purposes") | SKIPPED (user decision 2026-11-09; page confirm outstanding) — was: cleared pending user confirm | http://ipsar.fesb.unist.hr/HERIDAL%20database.html |
| 7 | LADI v1 | CC BY 4.0 (AWS Open Data Registry License field covers whole LADI dataset incl. v1; repo code MIT; DISTRIBUTION STATEMENT A) | CLEARED (cite registry access date) | https://registry.opendata.aws/ladi |
| 8 | FLAME 3 | CC BY 4.0 (DataPort Open-Access page, DOI 10.21227/w0mz-aq48) | SKIPPED (user decision 2026-11-09; IEEE account signup) — was: cleared | https://ieee-dataport.org/open-access/flame-3-multimodal-uav-rgb-and-thermal-image-dataset (DOI 10.21227/w0mz-aq48) |
| 9 | D-Fire | collection CC0 1.0 (repo LICENSE) BUT `PublicDataset*` images "remain subject to the terms of their original providers" (HPWREN/UCSD FIgLib, FESB Split wildfire DB credit-required, FireNet/Jadon et al. 2019) | CLEARED-WITH-CONDITIONS for research; for commercial/deployment filter to own-captured subset or clear each upstream + credit FESB | https://raw.githubusercontent.com/gaia-solutions-on-demand/DFireDataset/master/README.md |
| 10 | TREC-IS | no CC license; distributed as tweet-ID lists per incident, hydrate via twarc with own X API credentials; governed by TREC/NIST research use + X Developer Agreement (no full-text redistribution) | SKIPPED (user decision 2026-11-09; X API key + ToS) — was: cleared for research with constraints | http://trecis.org (topics/IDs; method per TREC-IS 2018 Guidelines PDF) |
| 11 | CrisisHelpOffer (101k tweets, arXiv:2502.16839) | none ("dataset will be provided upon request"; no public URL, no license; X-sourced so X ToS will apply) | BLOCKED until authors release — SKIPPED (user decision 2026-11-09; needs author email) | https://arxiv.org/abs/2502.16839 |

## 2. RescueNet ND and NC/SA disposition (AC3)

- **RescueNet — CC BY-NC-ND 4.0, confirmed primary**
  (BinaLab README "released under CC BY-NC-ND", downloads via
  Dropbox/figshare collection 6647354, paper s41597-023-02799-4).
  ND bars derivatives; finetuned weights are plausibly a derivative, so
  **BLOCKED for finetuning** (#7/#8 must not train on it). Permitted as
  **held-out EVAL ONLY** (inference, no weight updates), non-commercial,
  with attribution — cite from the #19 eval harness.
  Primary: https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation
- **NC/SA corpora — CLEARED for non-commercial research finetuning**,
  consistent with the Cohere CC-BY-NC base + LSAL non-commercial stack;
  **BLOCKED for commercial deployment.** xBD (CC BY-NC-SA), DisasterM3
  (CC-BY-NC-SA), MEDIC/CrisisMMD/HumAID (CC-BY-NC-SA; MEDIC HF API
  re-verified `cc-by-nc-sa-4.0`, ungated), SpaceNet 8 (CC-NC), FLAME 1
  (academic/non-commercial), DORA annotations (CC BY-NC-SA).
  SA rule: adapted dataset redistributions must carry the same license.
  Whether published model weights count as an SA derivative is legally
  gray — flag for legal before publishing weights trained on SA data.

## 3. URLs the user must personally sign / click through (cannot be cleared by an agent)

> **SKIPPED per user decision 2026-11-09.** All datasets below that require a
> personal signup, purchase, email, or click-through are out of scope until
> revisited. Rows in §1 carry a `SKIPPED` marker; §5 lists what remains
> usable. Exception: D-Fire's Kaggle mirror is available (`KAGGLE_API_TOKEN`
> provided), and its OneDrive links need no signup.

1. SARD — create IEEE account + buy individual subscription (~$40/month,
   min 1 month; free account covers Open Access only), accept DataPort
   ToS, download: https://ieee-dataport.org/subscribe — attach receipt +
   license screenshot to the issue.
2. FLAME 3 — create free IEEE/DataPort account and accept ToS at
   download: https://www.ieee.org/profile/public/createwebaccount/showCreateAccount.html?sourceCode=dataport
3. HERIDAL — open http://ipsar.fesb.unist.hr/HERIDAL%20database.html,
   confirm license string + whether download is direct or on-request,
   paste evidence.
4. TREC-IS — download topics/IDs from http://trecis.org (verify
   reachable), supply own X API key, accept X ToS; keep hydration log.
5. CrisisHelpOffer — email authors (contact via
   https://arxiv.org/abs/2502.16839 view-email) requesting data +
   explicit license/redistribution terms; paste reply.
6. Sen1Floods11 — email support@cloudtostreet.info (address in README)
   requesting a written license; paste reply into the issue.
7. DAD/CrisisNLP — open https://crisisnlp.qcri.org/terms-of-use.html and
   paste the exact text into the issue (download = clickwrap acceptance).
8. Kaggle mirror of D-Fire (optional; OneDrive links need no signup) —
   needs Kaggle account/ToS: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo

## 4. Verification log (issue test plan)

- Every row above carries a primary-source URL (§1, §2).
- Second-agent spot-check (this implementation, 6/5 rows, 2026-11-09):
  VIDI README fetched — zero "license" mentions, download via
  `youtube-dl` (no license stated, confirmed); Sen1Floods11 README
  fetched — zero "license" mentions, `support@cloudtostreet.info`
  contact (no license stated, confirmed); LADI AWS registry fetched —
  "Creative Commons Attribution 4.0 International (CC BY 4.0)"
  (confirmed); DMD UCI page fetched — "licensed under CC BY 4.0",
  DOI 10.24432/C52P6P (confirmed); D-Fire README fetched —
  `PublicDataset*` "subject to the terms of their original providers"
  (confirmed); RescueNet BinaLab repo page — "released under CC
  BY-NC-ND" (confirmed).
- Gated items still need user artifacts: DataPort subscription receipt +
  license screenshots, X API hydration log, pasted author /
  Cloud-to-Street emails, HERIDAL / trecis.org / CrisisNLP page captures.
- Agent-unverifiable from sandbox (network failures, need user or
  follow-up check): crisisnlp.qcri.org DNS failed (used search-cached
  primary text); ipsar.fesb.unist.hr TLS EOF (used search-cached primary
  text); trecis.org TLS EOF (used guidelines PDF + homepage snippets).
- Legal gray (needs-legal, not agent-decidable): weights-as-derivative
  under CC ND/SA; YouTube ToS + uploader-rights path for VIDI; X/Twitter
  redistribution limits for TREC-IS/CrisisHelpOffer/DAD images.

## 5. Training-availability summary for #7/#8

- Cleared, no signup needed: DMD, LADI v1, D-Fire (research;
  commercial needs subset filter; Kaggle mirror available via
  `KAGGLE_API_TOKEN`, OneDrive needs no signup), all NC/SA QCRI sets
  (research-only; prefer the MEDIC bundle over standalone DAD).
- SKIPPED per user decision 2026-11-09 (personal signup/purchase/email
  required): SARD, FLAME 3, HERIDAL, TREC-IS, DAD (terms capture
  outstanding), CrisisHelpOffer, Sen1Floods11.
- Blocked: CrisisHelpOffer (no release), VIDI + Sen1Floods11 (no
  license; VIDI also platform-ToS), RescueNet for training (ND;
  eval-only).
