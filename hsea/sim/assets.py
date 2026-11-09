"""Drone specification assets.

Vehicle capabilities (frame, endurance, sensors, autonomy skills) are
versioned YAML assets under ``assets/drones/``. Values flagged
``provenance: placeholder`` must be replaced by specs pulled from ArduPilot
sources (frame parameter files, ardupilot_gazebo models) in the assets
workstream; see docs/assets.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "drones"

REQUIRED_KEYS = (
    "name",
    "vehicle_class",
    "provenance",
    "capabilities",
)


@dataclass(frozen=True)
class DroneSpec:
    name: str
    vehicle_class: str
    provenance: str | dict
    raw: dict = field(compare=False)

    @property
    def sourced(self) -> bool:
        """True once placeholder values were replaced by ArduPilot pulls."""
        return self.provenance != "placeholder"

    @property
    def capabilities(self) -> dict:
        return dict(self.raw.get("capabilities", {}))

    @property
    def sensors(self) -> list[str]:
        return list(self.raw.get("sensors", []))

    @property
    def sitl_params(self) -> dict:
        return dict(self.raw.get("sitl_params", {}))

    @property
    def gazebo_model(self) -> str | None:
        return self.raw.get("gazebo_model")

    def supports(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, False))


def load_drone_spec(name: str, assets_dir: str | Path = ASSETS_DIR) -> DroneSpec:
    path = Path(assets_dir) / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Drone spec {name} missing keys: {missing}")
    return DroneSpec(
        name=raw["name"],
        vehicle_class=raw["vehicle_class"],
        provenance=raw["provenance"],
        raw=raw,
    )


# Capabilities the drone_mission solver needs from the assigned airframe.
MISSION_REQUIRED_CAPABILITIES = (
    "guided_takeoff",
    "guided_goto",
    "loiter",
    "rtl",
)


def assert_mission_capable(spec: DroneSpec) -> None:
    """Fail loudly when the airframe cannot fly bench missions."""
    missing = [c for c in MISSION_REQUIRED_CAPABILITIES if not spec.supports(c)]
    if missing:
        raise ValueError(
            f"Drone spec {spec.name!r} lacks mission capabilities: {missing}"
        )


def list_drone_specs(assets_dir: str | Path = ASSETS_DIR) -> list[str]:
    return sorted(p.stem for p in Path(assets_dir).glob("*.yaml"))
