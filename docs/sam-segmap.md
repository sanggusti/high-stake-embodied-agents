# SAM Segmentation Maps → Routes + Severity

Drone footage → Meta SAM 3.1 (`sam-3.1` on the Model API Responses
endpoint) → per-concept masks → route blockages + severity for the
recon advisor. One request per concept; the LM never sees pixels, only
the seg-map digest line in edge reports.

## Pipeline

```
native frame (Gazebo / Colosseum capture_frame)
  -> build_seg_map: one request per scenario concept (road, water, ...)
  -> parse boxes+masks, decode rasters (meta_sam_parser)
  -> rasterize to a 64x64 frame grid
  -> apply_overlap_rules  => flooded_road, smoke_blocked_road
  -> severity_of          => minor / moderate / severe
  -> seg_summary_line     => "segmap severity=severe(0.36) ..."
```

## Concept table (evidence from the 2026-11-02 live baseline)

Base concepts on every frame: `road`, `building`, `person`, `vehicle`,
`car`, `animal` (`animal` is oracle-covered, unbaselined on footage).
Scenario extras: fire → `smoke`; tsunami/flash_flood/flood → `water`;
earthquake → `tent` (relief camps); landslide → none yet.

Live scans are capped by `DMB_SAM_MAX_SCANS_PER_EPISODE` (default 3):
past the budget the edge reuses its last seg-map and marks reports
stale, bounding episode cost at ~7 concepts × budget × $0.0025.

Proven (26 image requests, true-mask overlays checked): houses ×41–50,
roads through rubble, flooded-region masks (frame 0.36), water, person,
car, smoke plume (0.23). Video: 90/90 frames, 16/16 houses tracked with
stable IDs in 7.4 s.

Zero-match (do not prompt these): `flooded street`, `damaged road`,
`rubble`, `debris`, `pile`, `collapsed/destroyed building`, `concrete`,
`fire` (smoke-obscured), `victim` on sim renders. Rule: prompt plain
concrete nouns, derive state in app code via mask overlap.

## Rubble fallback

Rubble-class concepts all return zero, but point/box prompts delineate
rubble precisely (point + negative-road prompt → mask at conf 0.906).
`rubble_point_prompt()` builds that prompt; candidate points come from
road-mask gaps or operator clicks (wiring is later work).

## Limits

- Native photorealistic frames only. MockBridge map renders return zero
  matches, so `SAMEdgePerception` degrades to empty reports on mock —
  route-map evals must run on gazebo-sitl/colosseum.
- Victim grounding needs a pixel→grid camera model (`grounding`
  callable); until it lands, `person` detections are reported only when
  a grounding is injected. Severity/blockages need no grounding.

## Knobs and cost

`DMB_SAM_BASE_URL` (default `https://api.meta.ai/v1`), `DMB_SAM_API_KEY`
(falls back to `META_AI_API_KEY`), `DMB_SAM_MODEL` (default `sam-3.1`).
~$0.0025/image → ~$0.013/frame for 5–6 concepts; video $0.20/1k frames
(50-frame minimum). Snapshot budget caps per-episode spend.

## Grounding and routes

`dmb/perception/grounding.py` is exact pinhole math: mask pixel → ray →
ground plane → grid cell, given camera, drone pose, and grid scale.
`dmb/perception/segmap.py` turns grounded cells into a cost grid (road
0.5, open 1, hazard 10, blocked 25) and A*-plans one route per located
victim. The `request_route_map` tool (present only with SAM perception)
captures a fresh seg-map on a snapshot-budget token and returns
rescuer-facing waypoint text; routes to unlocated victims are never
given. No camera is calibrated yet: the tool uses the documented
`survey_camera` nadir stance and labels output assumed. Evidence and
figures: [sam-routes-report](sam-routes-report.md).

## Native A/B runbook (gated)

Compares `DMB_EDGE_PERCEPTION=stub` vs `sam` arms on native frames.
Gates: #45 keyframes green, #48 footage flowing, camera FOV/pose logged
for calibration, scan budget set. Same seeds both arms:

```bash
DMB_SIM_BACKEND=gazebo-sitl <log-vars> \
  .venv/bin/inspect eval evals/forest_fire.py --model <id> --limit N
# rerun with DMB_EDGE_PERCEPTION=sam, then:
.venv/bin/python -m dmb.aggregate record --run-id <stub> ... --log <stub.eval>
.venv/bin/python -m dmb.aggregate record --run-id <sm> ... --log <sam.eval>
.venv/bin/python -m dmb.aggregate uplift --base <stub> --sam <sm>
```

Label every native route output with its geometry provenance (assumed
until calibrated). Oracle ceiling first: `evals/routes_oracle.py`
(scripted, provider-free) must read excess 1.0 / safety 1.0 on clean
worlds before any live A/B counts.

## Module map

- `dmb/perception/sam.py`: parser, masks, `concepts_for_scenario`,
  overlap rules, severity, `build_seg_map`, `sam_from_env`,
  `SAMEdgePerception` (same triggers/budget as the rule stub).
- `dmb/perception/grounding.py`: `CameraModel`, `project_pixel`,
  `project_mask`, assumed `survey_camera`.
- `dmb/perception/segmap.py`: cost grid, A* `plan_route`,
  `format_route_map`, shared `render_route_map`.
- `dmb/perception/oracle.py`: perfect seg-maps from sim truth (ceiling
  evals only); `evals/routes_oracle.py` + `tests/test_oracle_routes.py`.
- `tests/test_sam.py`, `tests/test_grounding.py`, `tests/test_segmap.py`
  + `tests/fixtures/sam_*.txt` (real outputs).
