# FIRMS fire archive (NASA)

Source: NASA FIRMS VIIRS NRT/Archive, downloaded 2026-11 as
`DL_FIRE_SV-C2_836814.zip` (Sep-Oct 2019, Southeast Asia).

* The **zip + Readme are committed**; the extracted 30 MB CSV is
  git-ignored and regenerates with:
  `unzip -o data/firm/DL_FIRE_SV-C2_836814.zip -d data/firm`
* Scenario ignitions are **baked** into `scenarios/firms_peat_*.yaml`
  (see `dmb/data/firms.py`); nothing at runtime or test time reads
  this CSV (tests use `tests/fixtures/firms_sample.csv`).
* Provenance for the three peat events (anchors, t0, FRP stats):
  `docs/firms-sim2real/sim-to-real.md`.
