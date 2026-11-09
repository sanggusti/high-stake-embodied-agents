# Edge-First Perception

## Principle

Vision runs on the edge device (light, cheap, always-on). The hosted LM is
the **navigator/orchestrator**: it receives compact reports and spends a
small snapshot budget only when a decision needs pixels. This mirrors real
disaster deployments — scarce compute, scarce bandwidth, intermittent link —
and keeps LM scores about decisions, not detector quality.

## Report occasions

| Trigger | When | Content |
|---|---|---|
| `novelty` | Edge sees a never-reported detection | New labels + positions |
| `heartbeat` | Every `heartbeat_every_steps` env steps | In-view count, totals found |
| `requested` | LM calls `request_edge_report` | Full scan + snapshot if budget remains |
| (future) | `takeoff`, `hazard-proximity`, link-loss | Lifecycle/safety events |

## Budgets (env-configurable)

- `DMB_EDGE_REPORT_INTERVAL_STEPS` (default 5): heartbeat cadence.
- `DMB_EDGE_MAX_SNAPSHOTS_PER_EPISODE` (default 8): on-demand frames per
  episode. Exhaustion degrades to report-only, never fails the episode.

## Vision channel (#27)

- `request_edge_report` returns text + the observation frame image
  (`ContentImage`, detail low) while budget remains; frames persist in the
  inspect-ai eval log.
- Mock renders truth-annotated top-down maps; SITL serves native 640×480
  camera frames with telemetry overlay (rendered fallback if capture
  fails). Frames show only observable content: visible dynamics, found
  victims, drone pose — never unfound victims or invisible zones.
- `VLMEdgePerception` (`dmb/perception/vlm.py`) is the detector baseline:
  same triggers/budget, detections from a VLM over sensor-view frames
  (`DMB_VLM_BASE_URL` / `DMB_VLM_API_KEY` / `DMB_VLM_MODEL`). Native-pixel
  VLM input is later work (needs camera grounding).

## Implementation

- `dmb/perception/edge.py`: `EdgePerception` ABC, `TriggerConfig`,
  `FrameBudget`, `StubEdgePerception` (rule-based over `MockBridge` state).
- `dmb/perception/sam.py`: `SAMEdgePerception` — SAM 3.1 seg-maps
  (masks → route blockages + severity, same triggers/budget; native
  frames only, see [sam-segmap](sam-segmap.md)).
- Real backends implement the same interface with an on-device model; trigger
  semantics and budget accounting do not change.
- Snapshots are referenced by `frame_ref` artifact ids; the registry stores
  the mapping so any report can be audited against its pixels later.

## What is NOT scored here

Detector precision/recall belongs to a future edge-model eval workstream.
Mission scorers treat edge reports as given input and score what the LM did
with them (coverage, prioritisation, safety, efficiency).
