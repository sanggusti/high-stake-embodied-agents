"""Simulation backends behind one interface.

Track 1 (phase 1): Gazebo + ArduPilot SITL.
Track 2 (fidelity): Colosseum/Unreal (runs on the operator's GPU workstation).
``MockBridge`` is a deterministic grid world so evals, tools, and scorers run
anywhere with no simulator installed.
"""

from __future__ import annotations

import math
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dmb.perception.edge import MOBILITY_RADIUS_FACTOR


@dataclass
class DroneState:
    x: int = 0
    y: int = 0
    altitude_m: float = 0.0
    armed: bool = False
    airborne: bool = False
    battery_pct: float = 100.0


@dataclass
class Victim:
    category: str
    x: int
    y: int
    mobility: str = "static"
    found: bool = False
    rescued: bool = False
    found_step: int = -1


@dataclass
class Observation:
    text: str
    telemetry: dict
    edge_report: dict | None = None


def danger_cells_from_state(state: dict) -> set[tuple[int, int]]:
    """Danger cells realized by episode end (pure mirror of _in_hazard).

    Union of activated `cell_danger` entries (due step <= final steps)
    and phase-active static hazard circles. Shared by scorers and the
    perception oracle so both judge against identical truth.
    """
    steps = int(state.get("steps", 0) or 0)
    danger: set[tuple[int, int]] = set()
    for key, due in (state.get("cell_danger", {}) or {}).items():
        try:
            x, y = (int(v) for v in str(key).split(","))
            due_step = int(due)
        except (ValueError, TypeError):
            continue
        if due_step <= steps:
            danger.add((x, y))
    grid = state.get("grid", {}) or {}
    width, height = int(grid.get("width", 10)), int(grid.get("height", 10))
    phases = [str(p.get("name")) for p in state.get("phases", [])]
    try:
        phase_idx = phases.index(str(state.get("phase", "")))
    except ValueError:
        phase_idx = 0
    for hz in state.get("hazards", []) or []:
        try:
            if int(hz.get("active_from_phase", 0)) > phase_idx:
                continue
            cx, cy = int(hz.get("x", -1)), int(hz.get("y", -1))
            r = float(hz.get("radius", 0))
        except (ValueError, TypeError):
            continue
        for x in range(max(0, int(cx - r)), min(width, int(cx + r) + 1)):
            for y in range(max(0, int(cy - r)), min(height, int(cy + r) + 1)):
                if math.hypot(x - cx, y - cy) <= r:
                    danger.add((x, y))
    return danger


class SimBridge(ABC):
    """Backend-agnostic episode interface. One instance per sample."""

    @abstractmethod
    def reset(self, metadata: dict) -> Observation:
        """Load (scenario, seed) from sample metadata; return initial obs."""

    @abstractmethod
    def act(self, action: str, params: dict) -> Observation:
        """Apply one primitive; return the resulting observation."""

    @abstractmethod
    def get_state(self) -> dict:
        """Full serialisable episode state for scorers and the registry."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources (sim instance, connections)."""

    def capture_frame(self, ref: str) -> bytes | None:
        """PNG bytes for an observation frame, or None if unavailable."""
        return None


class MockBridge(SimBridge):
    """Deterministic grid world. No external dependencies."""

    # Vehicle classes the backend can honour; mismatch fails loudly at reset.
    SUPPORTED_VEHICLE_CLASSES = frozenset({"multicopter", "fixed-wing"})

    def __init__(self, step_budget: int = 60, battery_drain_per_step: float = 1.0) -> None:
        self.step_budget = step_budget
        self.battery_drain = battery_drain_per_step
        self.drones = [DroneState()]
        self.drone = self.drones[0]
        self._launch_spots = [(0, 0)]
        self.victims: list[Victim] = []
        self.hazards: list[dict] = []
        self.phases: list[dict] = []
        self.width = 10
        self.height = 10
        self.steps = 0
        self.sim_time = 0.0
        self.safety_violations = 0
        self.seed = 0
        self.scenario = "mock"
        self._announced_phase: str | None = None
        self._visited: set[tuple[int, int]] = set()
        self.first_find_step: int | None = None

    @classmethod
    def from_metadata(cls, metadata: dict, **kwargs: object) -> "MockBridge":
        bridge = cls(**kwargs)  # type: ignore[arg-type]
        bridge.reset(metadata)
        return bridge

    def reset(self, metadata: dict) -> Observation:
        from dmb.sim.assets import load_drone_spec  # noqa: PLC0415

        spec_name = str(metadata.get("drone_spec", "quad_iris_class"))
        spec = load_drone_spec(spec_name)
        if spec.vehicle_class not in self.SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"{type(self).__name__} cannot honour vehicle class "
                f"{spec.vehicle_class!r} (spec {spec_name!r})"
            )
        self.drone_spec_name = spec_name
        grid = metadata.get("grid", {})
        self.width = int(grid.get("width", 10))
        self.height = int(grid.get("height", 10))
        self.scenario = str(metadata.get("scenario", "mock"))
        self.seed = int(metadata.get("seed", 0))
        self._sample_id = str(
            metadata.get("sample_id") or f"{self.scenario}-seed-{self.seed}"
        )
        self._weather_active = metadata.get("weather_active")
        # Actor schema (#19): unknown mobility fails loudly at reset.
        for v in metadata.get("victim_placements", []):
            if v.get("mobility", "static") not in MOBILITY_RADIUS_FACTOR:
                raise ValueError(
                    f"Unknown victim mobility {v.get('mobility')!r} "
                    f"(have {sorted(MOBILITY_RADIUS_FACTOR)})"
                )
        self.victims = [Victim(**v) for v in metadata.get("victim_placements", [])]
        self.hazards = list(metadata.get("hazards", []))
        self.phases = list(metadata.get("difficulty", {}).get("phases", []))
        fleet_size = int(metadata.get("fleet_size", 1))
        if fleet_size < 1:
            raise ValueError(f"fleet_size must be >= 1 (got {fleet_size})")
        corners = [
            (0, 0),
            (self.width - 1, 0),
            (0, self.height - 1),
            (self.width - 1, self.height - 1),
        ]
        self._launch_spots = [corners[i % len(corners)] for i in range(fleet_size)]
        self.drones = [
            DroneState(x=lx, y=ly, battery_pct=100.0)
            for lx, ly in self._launch_spots
        ]
        self.drone = self.drones[0]
        self.steps = 0
        self.sim_time = 0.0
        self.safety_violations = 0
        self._announced_phase = None
        self._visited = set(self._launch_spots)
        self.first_find_step = None
        self._init_dynamics(metadata.get("dynamics", {}))
        if fleet_size == 1:
            return self._observe("Episode start. Drone on pad at (0, 0), disarmed.")
        pads = ", ".join(f"#{i}@({x},{y})" for i, (x, y) in enumerate(self._launch_spots))
        return self._observe(f"Episode start. Fleet of {fleet_size} on pads: {pads}.")

    # -- primitives -----------------------------------------------------
    def _tick(self) -> None:
        self.steps += 1
        # Nominal 5 sim-seconds per action on the mock backend; SITL maps
        # wall time. Watcher `t` fields use this clock.
        self.sim_time += 5.0
        # Every airborne drone burns: actions are sequential in the log
        # but simultaneous in wall time (parallel-time model).
        for drone in self.drones:
            if drone.airborne:
                drone.battery_pct = max(0.0, drone.battery_pct - self.battery_drain)
        self._step_dynamics()

    def _init_dynamics(self, config: dict) -> None:
        from dmb.physics.fire import FireModel  # noqa: PLC0415
        from dmb.physics.terrain import SlipModel  # noqa: PLC0415
        from dmb.physics.water import WaterModel  # noqa: PLC0415

        # Unknown dynamics keys fail loudly: a typo'd block must never
        # silently disable a disaster's physics.
        unknown = set(config) - {"fire", "water", "terrain"}
        if unknown:
            raise ValueError(f"Unknown dynamics blocks: {sorted(unknown)}")

        def _block(name: str) -> dict | None:
            if name not in config:
                return None
            block = dict(config[name])
            if self._weather_active is not None:
                block["weather_active"] = self._weather_active
            return block

        fire_cfg = _block("fire")
        self._fire = FireModel(fire_cfg, self.width, self.height) if fire_cfg else None
        self._water = (
            WaterModel(config["water"], self.width, self.height)
            if "water" in config
            else None
        )
        terrain_cfg = _block("terrain")
        self._slip = (
            SlipModel(terrain_cfg, self.width, self.height) if terrain_cfg else None
        )
        self._danger: dict[tuple[int, int], int] = {}
        if self._fire is not None:
            self._danger.update(self._fire.danger_steps())
        if self._water is not None:
            self._danger.update(self._water.danger_steps())

    def _step_dynamics(self) -> None:
        if self._fire is not None:
            self._fire.step(self.steps)
            self._danger.update(self._fire.danger_steps())
        if self._water is not None:
            self._water.step(self.steps)
            for victim in self.victims:
                self._water.drift(victim, self.steps)
        if self._slip is not None:
            debris = self._slip.step(self.steps)
            for cell in debris:
                self._danger.setdefault(cell, self.steps)
            for victim in self.victims:
                if (victim.x, victim.y) in self._slip.debris_cells:
                    victim.mobility = "buried"

    def current_phase_index(self) -> int:
        """Index into self.phases by step count; 0 when no phases defined."""
        for i, phase in enumerate(self.phases):
            if self.steps < int(phase.get("until_step", 0)):
                return i
        return max(0, len(self.phases) - 1)

    def current_phase(self) -> str:
        if not self.phases:
            return "nominal"
        return str(self.phases[self.current_phase_index()].get("name", "nominal"))

    def _in_hazard(self, x: int, y: int) -> bool:
        danger_step = self._danger.get((x, y))
        if danger_step is not None and danger_step <= self.steps:
            return True
        phase_idx = self.current_phase_index()
        for hz in self.hazards:
            if int(hz.get("active_from_phase", 0)) > phase_idx:
                continue
            cx, cy, r = int(hz.get("x", -1)), int(hz.get("y", -1)), float(hz.get("radius", 0))
            if math.hypot(x - cx, y - cy) <= r:
                return True
        return False

    @property
    def any_airborne(self) -> bool:
        """True when at least one fleet drone is flying."""
        return any(d.airborne for d in self.drones)

    def _drone(self, idx: int) -> DroneState:
        if not 0 <= idx < len(self.drones):
            raise IndexError(f"drone_idx {idx} out of range (fleet {len(self.drones)})")
        return self.drones[idx]

    def arm(self, idx: int = 0) -> Observation:
        self._drone(idx).armed = True
        return self._observe("Armed.", idx)

    def takeoff(self, altitude_m: float, idx: int = 0) -> Observation:
        drone = self._drone(idx)
        if not drone.armed:
            return self._observe("Refused: arm before takeoff.", idx)
        if altitude_m <= 0:
            return self._observe("Refused: altitude must be positive.", idx)
        drone.airborne = True
        drone.altitude_m = altitude_m
        self._tick()
        return self._observe(f"Takeoff to {altitude_m:.1f} m.", idx)

    def goto(self, x: int, y: int, idx: int = 0) -> Observation:
        drone = self._drone(idx)
        if not drone.airborne:
            return self._observe("Refused: not airborne.", idx)
        drone.x = max(0, min(self.width - 1, x))
        drone.y = max(0, min(self.height - 1, y))
        self._visited.add((drone.x, drone.y))
        self._tick()
        if self._in_hazard(drone.x, drone.y):
            self.safety_violations += 1
            return self._observe(
                f"Moved to ({drone.x}, {drone.y}). WARNING: hazard zone.", idx
            )
        return self._observe(f"Moved to ({drone.x}, {drone.y}).", idx)

    def loiter(self, seconds: int = 5, idx: int = 0) -> Observation:
        drone = self._drone(idx)
        if not drone.airborne:
            return self._observe("Refused: not airborne.", idx)
        self._tick()
        return self._observe(f"Loitering {seconds}s at ({drone.x}, {drone.y}).", idx)

    def land(self, idx: int = 0) -> Observation:
        drone = self._drone(idx)
        drone.airborne = False
        drone.altitude_m = 0.0
        self._tick()
        return self._observe(f"Landed at ({drone.x}, {drone.y}).", idx)

    def rtl(self, idx: int = 0) -> Observation:
        drone = self._drone(idx)
        drone.x, drone.y = self._launch_spots[idx]
        drone.airborne = False
        drone.altitude_m = 0.0
        self._tick()
        return self._observe("Returned to launch and landed.", idx)

    # -- interface ------------------------------------------------------
    def act(self, action: str, params: dict) -> Observation:
        params = params or {}
        try:
            idx = int(params.get("drone_idx", 0))
            self._drone(idx)
        except (ValueError, IndexError):
            return self._observe(f"Unknown drone_idx: {params.get('drone_idx')!r}")
        handler = {
            "arm": lambda p: self.arm(idx),
            "takeoff": lambda p: self.takeoff(float(p.get("altitude_m", 10.0)), idx),
            "goto": lambda p: self.goto(int(p["x"]), int(p["y"]), idx),
            "loiter": lambda p: self.loiter(int(p.get("seconds", 5)), idx),
            "land": lambda p: self.land(idx),
            "rtl": lambda p: self.rtl(idx),
        }.get(action)
        if handler is None:
            return self._observe(f"Unknown action: {action}")
        return handler(params)

    def victims_within(self, radius: float) -> list[Victim]:
        return [
            v
            for v in self.victims
            if math.hypot(v.x - self.drone.x, v.y - self.drone.y) <= radius
        ]

    def mark_found(self, victim: Victim) -> None:
        victim.found = True
        if victim.found_step < 0:
            victim.found_step = self.steps
        if self.first_find_step is None:
            self.first_find_step = self.steps

    def _observe(self, event: str, idx: int = 0) -> Observation:
        phase = self.current_phase()
        if phase != self._announced_phase:
            self._announced_phase = phase
            event = f"{event} [phase: {phase}]"
        drone = self.drones[idx] if 0 <= idx < len(self.drones) else self.drone
        telemetry = {
            "drone_idx": idx,
            "x": drone.x,
            "y": drone.y,
            "altitude_m": drone.altitude_m,
            "armed": drone.armed,
            "airborne": drone.airborne,
            "battery_pct": round(drone.battery_pct, 1),
            "steps": self.steps,
            "safety_violations": self.safety_violations,
            "phase": phase,
        }
        return Observation(text=event, telemetry=telemetry)

    def _danger_kind(self) -> str:
        if getattr(self, "_fire", None) is not None:
            return "fire"
        if getattr(self, "_water", None) is not None:
            return "water"
        if getattr(self, "_slip", None) is not None:
            return "terrain"
        return "fire"

    @property
    def survey_camera(self):  # -> SurveyFrame | None (render_frames)
        """Pre-rendered survey frame + camera, if one covers this episode.

        Cached on first capture; None keeps the legacy map render.
        """
        return getattr(self, "_survey", None)

    @property
    def burning_cells(self) -> list[tuple[int, int]]:
        """Burning or smoldering cells (fire model), sorted, deterministic.

        Peat ignitions start underground (smolder) with surface breakout
        later; both stages count as fire for point-prompt seeding.
        """
        fire = getattr(self, "_fire", None)
        ignited = getattr(fire, "ignited", {}) or {}
        smolder = getattr(fire, "smolder", {}) or {}
        return sorted(set(ignited) | set(smolder))

    def capture_frame(self, ref: str) -> bytes | None:
        """PNG bytes for an observation frame, or None if unavailable."""
        from dmb.perception.frames import render_map_frame  # noqa: PLC0415
        from dmb.perception.render_frames import survey_frame  # noqa: PLC0415

        key = (self.scenario, self.seed)
        if getattr(self, "_survey_key", None) != key:
            self._survey = survey_frame(self.scenario, self.seed)
            self._survey_key = key
        if self._survey is not None:
            return self._survey.png
        state = self.get_state()
        state["danger_kind"] = self._danger_kind()
        return render_map_frame(state, {"width": self.width, "height": self.height})

    def get_state(self) -> dict:
        cells = max(1, self.width * self.height)
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "grid": {"width": self.width, "height": self.height},
            "steps": self.steps,
            "sim_time": self.sim_time,
            "step_budget": self.step_budget,
            "phase": self.current_phase(),
            "safety_violations": self.safety_violations,
            "coverage": len(self._visited) / cells,
            "first_find_step": self.first_find_step if self.first_find_step is not None else -1,
            "drone": {
                "x": self.drone.x,
                "y": self.drone.y,
                "altitude_m": self.drone.altitude_m,
                "armed": self.drone.armed,
                "airborne": self.drone.airborne,
                "battery_pct": self.drone.battery_pct,
            },
            "fleet_size": len(self.drones),
            "drones": [
                {
                    "x": d.x,
                    "y": d.y,
                    "altitude_m": d.altitude_m,
                    "armed": d.armed,
                    "airborne": d.airborne,
                    "battery_pct": d.battery_pct,
                }
                for d in self.drones
            ],
            "victims": [
                {
                    "category": v.category,
                    "x": v.x,
                    "y": v.y,
                    "mobility": v.mobility,
                    "found": v.found,
                    "rescued": v.rescued,
                    "found_step": v.found_step,
                }
                for v in self.victims
            ],
            "hazards": self.hazards,
            "phases": self.phases,
            "cell_danger": {f"{x},{y}": s for (x, y), s in self._danger.items()},
            "dynamics": self._dynamics_summary(),
        }

    def _dynamics_summary(self) -> dict:
        """Light per-model counts for per-kind containment (#4)."""
        summary: dict = {}
        fire = getattr(self, "_fire", None)
        water = getattr(self, "_water", None)
        slip = getattr(self, "_slip", None)
        if fire is not None:
            summary["fire"] = {
                "ignited_cells": len(fire.ignited),
                "burned_cells": len(fire.burned_cells(self.steps)),
            }
        if water is not None:
            summary["water"] = {"flooded_cells": len(water._flooded)}
        if slip is not None:
            summary["terrain"] = {
                "debris_cells": len(slip.debris_cells),
                "slip_step": slip.slip_step if slip.slip_step is not None else -1,
            }
        return summary

    def close(self) -> None:
        return None


EARTH_RADIUS_M = 6378137.0


def _offset_latlon(lat0: float, lon0: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Equirectangular offset (metres) from an anchor. Good for <1 km."""
    lat = lat0 + (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon = lon0 + (east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return lat, lon


def _offset_meters(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """Inverse of _offset_latlon: (east, north) metres from anchor."""
    north_m = (lat - lat0) * (math.pi / 180.0) * EARTH_RADIUS_M
    east_m = (lon - lon0) * (math.pi / 180.0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return east_m, north_m


def _require_pymavlink():
    try:
        from pymavlink import mavutil  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "GazeboSITLBridge needs pymavlink: pip install -e '.[sim]'"
        ) from exc
    return mavutil


class GazeboSITLBridge(MockBridge):
    """Track 1 backend: one Docker worker (Gazebo + ArduPilot SITL) per episode.

    Flight semantics mirror MockBridge (grid, victims, phases, scoring) but
    every primitive flies for real over MAVLink. Grid cells map to metres
    around the home anchor (east = +x, north = +y).

    Perception in #2 is truth-based (same edge stub as mock): victim
    coordinates come from scenario metadata, found when the drone's real
    position is within radius. Camera perception arrives with watcher T2.
    """

    # Runs ArduCopter SITL + the iris Gazebo world: multicopters only.
    SUPPORTED_VEHICLE_CLASSES = frozenset({"multicopter"})

    def __init__(
        self,
        image: str = "dmb/sim:track1",
        grid_cell_m: float = 5.0,
        boot_timeout_s: float = 240.0,
        action_timeout_s: float = 60.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.image = image
        self.grid_cell_m = grid_cell_m
        self.boot_timeout_s = boot_timeout_s
        self.action_timeout_s = action_timeout_s
        self._container_id: str | None = None
        self._mav = None
        self._home: tuple[float, float] | None = None
        self._latest: dict = {}

    # -- lifecycle ------------------------------------------------------
    def reset(self, metadata: dict) -> Observation:
        obs = super().reset(metadata)
        self._start_worker()
        try:
            self._connect_and_prepare()
        except Exception:
            self.close()
            raise
        note = (
            f"SITL worker {self._container_id[:12]} ready; "
            f"home grid maps {self.grid_cell_m}m/cell."
        )
        return Observation(text=f"{obs.text} {note}", telemetry=self._telemetry())

    def close(self) -> None:
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None
        if self._container_id is not None:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True, timeout=60,
            )
            self._container_id = None

    @property
    def container_id(self) -> str | None:
        return self._container_id

    def capture_frame(self, ref: str) -> bytes | None:
        """Native camera frame with telemetry overlay; rendered fallback."""
        from dmb.perception.frames import GazeboFrameCapture, overlay_telemetry  # noqa: PLC0415

        if self._container_id is not None and getattr(self, "_partition", None):
            grab = GazeboFrameCapture(self._container_id, self._partition)
            png = grab.capture(ref)
            if png is not None:
                state = self.get_state()
                drone = state["drone"]
                return overlay_telemetry(
                    png,
                    [
                        f"{state['scenario']} step={state['steps']} phase={state['phase']}",
                        f"pos=({drone['x']},{drone['y']}) alt={drone['altitude_m']}m batt={drone['battery_pct']}%",
                    ],
                )
        return super().capture_frame(ref)

    def _docker_run_args(self) -> list[str]:
        """docker run argv; L3 recording mount is opt-out (DMB_L3=off)."""
        from dmb.watch.l3 import l3_enabled, l3_sample_dir  # noqa: PLC0415

        args = ["docker", "run", "-d", "--rm", "-e", f"GZ_PARTITION={self._partition}"]
        self._l3_dir = None
        if l3_enabled():
            sample_id = getattr(self, "_sample_id", "unknown-sample")
            self._l3_dir = l3_sample_dir(sample_id)
            args += [
                "-v", f"{self._l3_dir}:/record",
                "-e", "DMB_GZ_RECORD_PATH=/record",
            ]
        args += ["-p", "127.0.0.1::5760", self.image]
        return args

    def _start_worker(self) -> None:
        # Unique transport partition: parallel workers must never merge
        # into one Gazebo discovery domain.
        self._partition = f"dmb-{uuid.uuid4().hex[:12]}"
        proc = subprocess.run(
            self._docker_run_args(), capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.strip()}")
        self._container_id = proc.stdout.strip()

    def _container_logs(self, tail: int = 30) -> str:
        if self._container_id is None:
            return "<no container>"
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self._container_id],
            capture_output=True, text=True, timeout=30,
        )
        return (proc.stdout + proc.stderr).strip() or "<empty logs>"

    def _worker_port(self) -> int:
        assert self._container_id is not None
        proc = subprocess.run(
            ["docker", "port", self._container_id, "5760"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"docker port failed: {proc.stderr.strip()}")
        return int(proc.stdout.strip().rsplit(":", 1)[-1])

    def _connect_and_prepare(self) -> None:
        mavutil = _require_pymavlink()
        deadline = time.time() + self.boot_timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            mav = None
            try:
                port = self._worker_port()
                mav = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}")
                # NOTE: wait_heartbeat returns None (no raise) on timeout or
                # dead socket — must check explicitly or we keep a dead link.
                if mav.wait_heartbeat(timeout=20) is None:
                    raise TimeoutError("no heartbeat from SITL worker")
                self._mav = mav
                mav = None  # owned by self now
                break
            except Exception as exc:  # container still booting
                last_err = exc
                if mav is not None:
                    # Never leave orphaned sockets: a stale half-open
                    # connection can steal or break the accepted slot.
                    try:
                        mav.close()
                    except Exception:
                        pass
                time.sleep(5)
        else:
            raise RuntimeError(f"SITL never answered: {last_err}")
        assert self._mav is not None
        self._mav.mav.request_data_stream_send(
            self._mav.target_system, self._mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1,
        )
        self._await_home()

    def _await_home(self, timeout_s: float = 180.0) -> None:
        assert self._mav is not None
        mavutil = _require_pymavlink()
        deadline = time.time() + timeout_s
        last_request = 0.0
        saw_streams = False
        while time.time() < deadline:
            if time.time() - last_request > 15:
                self._mav.mav.request_data_stream_send(
                    self._mav.target_system, self._mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1,
                )
                last_request = time.time()
            if self._recv("SYS_STATUS", timeout=5) is not None:
                saw_streams = True
            msg = self._recv("GLOBAL_POSITION_INT", timeout=5)
            if msg is not None and msg.lat != 0:
                self._home = (msg.lat / 1e7, msg.lon / 1e7)
                self._sync_from_telemetry()
                return
        hint = "streams flowing, no GPS (Gazebo/JSON link?)" if saw_streams else "no telemetry at all"
        raise RuntimeError(
            f"No GPS home fix from SITL worker ({hint}). "
            f"Container logs:\n{self._container_logs()}"
        )

    # -- messaging helpers ----------------------------------------------
    def _recv(self, msg_type: str, timeout: float = 5.0):
        assert self._mav is not None
        msg = self._mav.recv_match(type=msg_type, blocking=True, timeout=timeout)
        if msg is not None:
            self._latest[msg_type] = msg
        return self._latest.get(msg_type)

    def _set_mode(self, name: str, timeout: float = 15.0) -> bool:
        assert self._mav is not None
        target = self._mav.mode_mapping()[name]
        deadline = time.time() + timeout
        while time.time() < deadline and self._mav.flightmode != name:
            self._mav.set_mode(target)
            self._mav.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        return self._mav.flightmode == name

    def _grid_to_latlon(self, x: int, y: int) -> tuple[float, float]:
        assert self._home is not None
        lat0, lon0 = self._home
        return _offset_latlon(lat0, lon0, x * self.grid_cell_m, y * self.grid_cell_m)

    def _latlon_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        assert self._home is not None
        lat0, lon0 = self._home
        east_m, north_m = _offset_meters(lat0, lon0, lat, lon)
        return (
            max(0, min(self.width - 1, int(round(east_m / self.grid_cell_m)))),
            max(0, min(self.height - 1, int(round(north_m / self.grid_cell_m)))),
        )

    def _sync_from_telemetry(self) -> None:
        pos = self._recv("GLOBAL_POSITION_INT", timeout=5)
        if pos is not None and self._home is not None:
            gx, gy = self._latlon_to_grid(pos.lat / 1e7, pos.lon / 1e7)
            self.drone.x, self.drone.y = gx, gy
            self._visited.add((gx, gy))
            self.drone.altitude_m = pos.relative_alt / 1000.0
        status = self._recv("SYS_STATUS", timeout=5)
        if status is not None:
            self.drone.battery_pct = float(status.battery_remaining)
        if self._mav is not None:
            self.drone.armed = bool(self._mav.motors_armed())

    def _telemetry(self) -> dict:
        self._sync_from_telemetry()
        return {
            "x": self.drone.x,
            "y": self.drone.y,
            "altitude_m": round(self.drone.altitude_m, 1),
            "armed": self.drone.armed,
            "airborne": self.drone.airborne,
            "battery_pct": round(self.drone.battery_pct, 1),
            "steps": self.steps,
            "safety_violations": self.safety_violations,
            "phase": self.current_phase(),
        }

    # -- primitives (real flight) ---------------------------------------
    def arm(self, idx: int = 0) -> Observation:
        assert self._mav is not None
        if not self._set_mode("GUIDED"):
            return self._observe("Refused: GUIDED mode rejected.")
        # Fresh workers fail pre-arm checks (IMU/EKF settling) for a while:
        # re-command periodically and report the actual vehicle reason.
        deadline = time.time() + 120
        last_prearm = "unknown"
        last_cmd = 0.0
        while time.time() < deadline and not self._mav.motors_armed():
            if time.time() - last_cmd > 5:
                self._mav.arducopter_arm()
                last_cmd = time.time()
            msg = self._mav.recv_match(
                type=["HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=2
            )
            if msg is not None and msg.get_type() == "STATUSTEXT":
                text = str(msg.text)
                if "PreArm" in text or "Arm:" in text:
                    last_prearm = text
        self.drone.armed = bool(self._mav.motors_armed())
        if not self.drone.armed:
            return self._observe(f"Refused: arming failed ({last_prearm}).")
        return self._observe("Armed in GUIDED.")

    def takeoff(self, altitude_m: float, idx: int = 0) -> Observation:
        assert self._mav is not None
        mavutil = _require_pymavlink()
        if not self.drone.armed:
            return self._observe("Refused: arm before takeoff.")
        if altitude_m <= 0:
            return self._observe("Refused: altitude must be positive.")
        self._mav.mav.command_long_send(
            self._mav.target_system, self._mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, altitude_m,
        )
        deadline = time.time() + self.action_timeout_s
        alt = 0.0
        while time.time() < deadline:
            msg = self._recv("GLOBAL_POSITION_INT", timeout=5)
            if msg is None:
                continue
            alt = msg.relative_alt / 1000.0
            if alt >= altitude_m - 1.0:
                break
        self.drone.airborne = alt >= altitude_m - 1.0
        self._tick()
        self._sync_from_telemetry()
        if not self.drone.airborne:
            return self._observe(f"Takeoff incomplete: {alt:.1f}m.")
        return self._observe(f"Takeoff to {alt:.1f} m.")

    def goto(self, x: int, y: int, idx: int = 0) -> Observation:
        assert self._mav is not None
        mavutil = _require_pymavlink()
        if not self.drone.airborne:
            return self._observe("Refused: not airborne.")
        tx = max(0, min(self.width - 1, x))
        ty = max(0, min(self.height - 1, y))
        lat, lon = self._grid_to_latlon(tx, ty)
        mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        hold_alt = max(3.0, self.drone.altitude_m)
        deadline = time.time() + self.action_timeout_s
        arrived = False
        while time.time() < deadline:
            self._mav.mav.set_position_target_global_int_send(
                0, self._mav.target_system, self._mav.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mask, int(lat * 1e7), int(lon * 1e7), hold_alt,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            self._sync_from_telemetry()
            dist_cells = math.hypot(self.drone.x - tx, self.drone.y - ty)
            if dist_cells < 1.0:
                arrived = True
                break
            time.sleep(1)
        self._tick()
        self._sync_from_telemetry()
        if self._in_hazard(self.drone.x, self.drone.y):
            self.safety_violations += 1
            return self._observe(
                f"Moved to ({self.drone.x}, {self.drone.y}). WARNING: hazard zone."
            )
        status = "Arrived" if arrived else "En route"
        return self._observe(f"{status} ({self.drone.x}, {self.drone.y}).")

    def loiter(self, seconds: int = 5, idx: int = 0) -> Observation:
        if not self.drone.airborne:
            return self._observe("Refused: not airborne.")
        time.sleep(min(seconds, 30))
        self._tick()
        self._sync_from_telemetry()
        return self._observe(f"Loitered at ({self.drone.x}, {self.drone.y}).")

    def land(self, idx: int = 0) -> Observation:
        if not self._set_mode("LAND", timeout=15):
            return self._observe("Refused: LAND mode rejected.")
        deadline = time.time() + 90
        alt = 999.0
        while time.time() < deadline:
            msg = self._recv("GLOBAL_POSITION_INT", timeout=5)
            if msg is None:
                continue
            alt = msg.relative_alt / 1000.0
            if alt < 0.3:
                break
        self.drone.airborne = alt >= 0.3
        self.drone.altitude_m = max(0.0, alt)
        self._tick()
        self._sync_from_telemetry()
        return self._observe(f"Landed at ({self.drone.x}, {self.drone.y}).")

    def rtl(self, idx: int = 0) -> Observation:
        if not self._set_mode("RTL", timeout=15):
            return self._observe("Refused: RTL mode rejected.")
        deadline = time.time() + 120
        assert self._mav is not None
        while time.time() < deadline and self._mav.motors_armed():
            self._mav.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        self.drone.airborne = False
        self.drone.altitude_m = 0.0
        self._tick()
        self._sync_from_telemetry()
        return self._observe("Returned to launch and landed.")


class ColosseumBridge(SimBridge):
    """Track 2 backend: Colosseum/Unreal on the operator GPU workstation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "ColosseumBridge is implemented in the fidelity-track issue. "
            "Use MockBridge until then."
        )

    def reset(self, metadata: dict) -> Observation:
        raise NotImplementedError

    def act(self, action: str, params: dict) -> Observation:
        raise NotImplementedError

    def get_state(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
