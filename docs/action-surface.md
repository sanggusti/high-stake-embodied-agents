# Action Surface — ArduPilot-Derived Drone Tools

## Source of truth

Flight semantics follow ArduPilot Copter in **GUIDED mode**: `mode guided`
→ `arm throttle` → `takeoff <alt>` → position/loiter commands → `land`/`RTL`,
with missions and geofence available. `MockBridge` implements identical
tool-level semantics on a grid; `GazeboSITLBridge` flies them for real.

## MAVLink bindings (Track 1, verified against Copter-4.6.3 SITL)

Link: pymavlink TCP to the worker's port 5760, heartbeat + requested
`MAV_DATA_STREAM_ALL` telemetry. Grid cells map to metres around the GPS
home anchor (east = +x, north = +y, 5 m/cell).

| Tool | MAVLink | Notes |
|---|---|---|
| `arm` | `SET_MODE` GUIDED, then `MAV_CMD_COMPONENT_ARM_DISARM` | GUIDED first; arm polled via heartbeat flags |
| `takeoff` | `MAV_CMD_NAV_TAKEOFF` (param7 = alt) | complete when `GLOBAL_POSITION_INT.relative_alt` ≥ target − 1 m |
| `goto_location` | `SET_POSITION_TARGET_GLOBAL_INT`, `MAV_FRAME_GLOBAL_RELATIVE_ALT_INT`, position-only mask (3576) | re-sent at 1 Hz until within 1 cell; altitude holds current (≥ 3 m) |
| `loiter` | (none — timed hold) | GUIDED holds position; waits ≤ 30 s |
| `land` | `SET_MODE` LAND | complete when relative alt < 0.3 m |
| `return_to_launch` | `SET_MODE` RTL | complete on disarm |
| `get_telemetry` | `GLOBAL_POSITION_INT`, `SYS_STATUS`, `VFR_HUD`, `GPS_RAW_INT` | position → grid, battery %, alt |
| `request_edge_report` | (none — truth-based stub in #2) | victims from scenario metadata; camera frames arrive with watcher T2 |

Refusals (disarmed takeoff, grounded goto, rejected mode) return text, never
raise; only infrastructure failures (container, link loss) raise.

## LM tools (`dmb/tools/drone.py`)

| Tool | Args | Effect |
|---|---|---|
| `arm` | — | Arm motors (required before takeoff) |
| `takeoff` | `altitude_m: float` | Climb and hover; refused if disarmed or non-positive |
| `goto_location` | `x: int, y: int` | Fly to coordinates (grid now; lat/lon or local-NED on SITL) |
| `loiter` | `seconds: int` | Hold position while edge perception scans |
| `land` | — | Land in place |
| `return_to_launch` | — | Abort: return and land |
| `get_telemetry` | — | Position, altitude, battery, steps, violations |
| `request_edge_report` | — | On-demand perception scan (consumes snapshot budget) |
| `request_route_map` | — | SAM-only: fresh seg-map + ground routes to located victims (consumes snapshot budget) |

Movement tools append any edge report (`[edge:<trigger>] ...`) so perception
arrives as a side channel of acting, matching deployment reality.

## Higher-level skills (operator-named)

- **Non-GPS navigation**: composite skill over optical-flow / visual-odometry
  position estimates; exposed as a tool once the SITL message set is pinned.
  Mock behavior: allowed everywhere; GPS-denied zones arrive with Track 1.
- **Nav-mesh path planning**: `goto` through a mesh-aware planner instead of
  direct flight; mock moves directly, SITL/Colosseum route around obstacles.
- Both skills must degrade honestly: if the backend cannot do them, the tool
  errors instead of pretending.

## Tooling contract for new backends

1. One bridge instance per sample; `close()` releases everything.
2. `act()` returns an `Observation(text, telemetry)`; never raises on
   pilot error — refusals are text the LM can learn from.
3. `get_state()` is JSON-serialisable and scorer-sufficient.
4. Tool names and arg shapes stay stable across backends; only the
   coordinate frame and dynamics change (documented per backend).
