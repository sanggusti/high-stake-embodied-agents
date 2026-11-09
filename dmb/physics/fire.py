"""Cellular wildfire spread with FBP-style per-cell rate of spread.

ROS(cell <- burning neighbor) =
    base_ros * fuel * (1 - moisture) * wind_alignment(direction) * drought

Follows the Cell2Fire/FBP structure (per-cell fuel, weather, moisture,
topography over a homogeneous grid) minus calibrated fuel-type tables,
which are documented future work, not silent precision.

Peat extension (#50): an optional ``peat`` block adds an underground
smolder layer in the style of Indonesian peatland fires — slow flameless
creep through dried peat that breaks out to the surface at multiple
points, then runs fast under drought and produces thick haze. Without a
``peat`` block the model is bit-identical to the legacy surface model.
"""

from __future__ import annotations

import math

NEIGHBORS = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]


def _norm(vec: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(*vec)
    return (vec[0] / n, vec[1] / n) if n else (0.0, 0.0)


class FireModel:
    def __init__(self, config: dict, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.base_ros = float(config.get("base_ros", 0.25))
        self.threshold = float(config.get("ignition_threshold", 1.0))
        self.burn_duration = int(config.get("burn_duration", 12))
        self.moisture = float(config.get("moisture", 0.3))
        self.wind_speed = float(config.get("wind_speed", 0.0))
        # Weather coupling (#17): an explicit weather option overrides the
        # raw wind/moisture, so identical seeds diverge documentably across
        # weather. No weather key = raw config (legacy behavior).
        self.weather_active = config.get("weather_active")
        modifiers = config.get("weather_modifiers", {})
        if self.weather_active is not None:
            if self.weather_active not in modifiers:
                raise ValueError(
                    f"Unknown fire weather {self.weather_active!r} "
                    f"(have {sorted(modifiers)})"
                )
            mod = modifiers[self.weather_active]
            self.wind_speed = float(mod.get("wind_speed", self.wind_speed))
            self.moisture = float(mod.get("moisture", self.moisture))
        self.fuel_default = float(config.get("fuel_default", 0.7))
        self.fuel_map = {
            (int(c["x"]), int(c["y"])): float(c["fuel"])
            for c in config.get("fuel_map", [])
        }
        wind_dir = tuple(config.get("wind_dir", [1, 0]))
        self.wind = _norm((float(wind_dir[0]), float(wind_dir[1])))
        # Peat layer (#50). All defaults reproduce legacy behavior: no peat
        # anywhere (depth 0), drought neutral (1.0). Underground creep is
        # wind-independent; drought accelerates surface spread only.
        peat = config.get("peat") or {}
        self.peat_default = float(peat.get("depth_default", 0.0))
        self.peat_map = {}
        for block in peat.get("peat_blocks", []):
            for x in range(int(block["x0"]), int(block["x1"]) + 1):
                for y in range(int(block["y0"]), int(block["y1"]) + 1):
                    self.peat_map[(x, y)] = float(block["depth"])
        for c in peat.get("peat_map", []):
            self.peat_map[(int(c["x"]), int(c["y"]))] = float(c["depth"])
        self.smolder_ros = float(peat.get("smolder_ros", 0.1))
        self.smolder_threshold = float(peat.get("smolder_threshold", 0.5))
        self.breakout_threshold = float(peat.get("breakout_threshold", 1.0))
        self.drought = float(peat.get("drought", 1.0))
        if self.weather_active is not None:
            mod = modifiers.get(self.weather_active, {})
            if "drought" in mod:
                self.drought = float(mod["drought"])
        self.smolder_duration_mult = float(peat.get("smolder_duration_mult", 3.0))
        self.haze_per_burning = float(peat.get("haze_per_burning", 0.05))
        self.haze_per_smoldering = float(peat.get("haze_per_smoldering", 0.08))
        self.ignited: dict[tuple[int, int], int] = {}
        self.exposure: dict[tuple[int, int], float] = {}
        self.smolder: dict[tuple[int, int], int] = {}
        self.smolder_exposure: dict[tuple[int, int], float] = {}
        for cell in config.get("ignition", []):
            key = (int(cell["x"]), int(cell["y"]))
            if not (0 <= key[0] < width and 0 <= key[1] < height):
                continue
            if self.peat_at(key) > 0:
                # Ignition on peat starts underground; the surface
                # breakout follows once smolder exposure matures.
                self.smolder[key] = 0
                self.smolder_exposure[key] = self.smolder_threshold
            else:
                self.ignited[key] = 0

    def fuel_at(self, cell: tuple[int, int]) -> float:
        return self.fuel_map.get(cell, self.fuel_default)

    def peat_at(self, cell: tuple[int, int]) -> float:
        return self.peat_map.get(cell, self.peat_default)

    def smolder_duration(self) -> int:
        return int(self.burn_duration * self.smolder_duration_mult)

    def wind_alignment(self, direction: tuple[int, int]) -> float:
        """Downwind spread faster; upwind never fully blocked (floor 0.2)."""
        dx, dy = _norm((float(direction[0]), float(direction[1])))
        dot = dx * self.wind[0] + dy * self.wind[1]
        return max(0.2, 1.0 + dot * self.wind_speed * 0.15)

    def burning_cells(self, step: int) -> set[tuple[int, int]]:
        return {
            cell for cell, start in self.ignited.items()
            if start <= step < start + self.burn_duration
        }

    def burned_cells(self, step: int) -> set[tuple[int, int]]:
        return {
            cell for cell, start in self.ignited.items()
            if step >= start + self.burn_duration
        }

    def smoldering_cells(self, step: int) -> set[tuple[int, int]]:
        duration = self.smolder_duration()
        return {
            cell for cell, start in self.smolder.items()
            if start <= step < start + duration
        }

    def haze_index(self, step: int) -> float:
        """0-1 smoke loading: peat smolder contributes thicker haze per cell
        than surface flame. Drives UE fog density (T2); edge visibility
        coupling is documented follow-up, not wired here."""
        haze = (
            self.haze_per_burning * len(self.burning_cells(step))
            + self.haze_per_smoldering * len(self.smoldering_cells(step))
        )
        return min(1.0, round(haze, 4))

    def step(self, current_step: int) -> list[tuple[int, int]]:
        """Advance one step; return newly surface-ignited cells."""
        new: list[tuple[int, int]] = []
        for cell in self.burning_cells(current_step):
            for dx, dy in NEIGHBORS:
                nb = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nb[0] < self.width and 0 <= nb[1] < self.height):
                    continue
                if nb in self.ignited:
                    continue
                ros = (
                    self.base_ros
                    * self.fuel_at(nb)
                    * (1.0 - self.moisture)
                    * self.wind_alignment((dx, dy))
                    * self.drought
                )
                self.exposure[nb] = self.exposure.get(nb, 0.0) + ros
                if self.exposure[nb] >= self.threshold:
                    self.ignited[nb] = current_step
                    new.append(nb)
        self._step_peat(current_step)
        return new

    def _step_peat(self, current_step: int) -> None:
        """Underground creep (wind-independent) plus surface breakouts."""
        if self.peat_default <= 0 and not self.peat_map:
            return
        dry = 1.0 - self.moisture
        # Smoldering cells self-feed toward breakout and creep outward;
        # burning cells light the peat beside and beneath them.
        sources = self.smoldering_cells(current_step) | self.burning_cells(current_step)
        for cell in sources:
            for dx, dy in NEIGHBORS:
                nb = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nb[0] < self.width and 0 <= nb[1] < self.height):
                    continue
                depth = self.peat_at(nb)
                if depth <= 0 or nb in self.smolder:
                    continue
                gain = self.smolder_ros * depth * dry
                self.smolder_exposure[nb] = self.smolder_exposure.get(nb, 0.0) + gain
                if self.smolder_exposure[nb] >= self.smolder_threshold:
                    self.smolder[nb] = current_step
        for cell in self.smoldering_cells(current_step):
            if cell in self.ignited:
                continue
            gain = self.smolder_ros * self.peat_at(cell) * dry * self.drought
            self.smolder_exposure[cell] = self.smolder_exposure.get(cell, 0.0) + gain
            if (
                self.smolder_exposure[cell]
                >= self.smolder_threshold + self.breakout_threshold
            ):
                self.ignited[cell] = current_step

    def danger_steps(self) -> dict[tuple[int, int], int]:
        """Cell -> step it became dangerous (surface ignition).

        Smoldering peat is underground: it drives haze and later
        breakouts but is not itself a flight hazard."""
        return dict(self.ignited)

    def snapshot(self, step: int | None = None) -> dict:
        if step is None:
            starts = list(self.ignited.values()) + list(self.smolder.values())
            step = max(starts) if starts else 0
        return {
            "weather": self.weather_active,
            "wind_speed": self.wind_speed,
            "moisture": self.moisture,
            "ignited": sorted(f"{x},{y}@{s}" for (x, y), s in self.ignited.items()),
            "smolder": sorted(f"{x},{y}@{s}" for (x, y), s in self.smolder.items()),
            "haze": self.haze_index(step),
        }
