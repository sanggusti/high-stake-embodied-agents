"""Landslide: slope-stability rule with saturation trigger (T0).

Soil saturation rises every step (faster when raining). When it crosses
the threshold, the steepest candidate cell slips; debris runs along the
configured direction, burying victims in its path. One slip per episode
unless configured otherwise. Deterministic.
"""

from __future__ import annotations


class SlipModel:
    def __init__(self, config: dict, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.candidates = [
            {"x": int(c["x"]), "y": int(c["y"]), "slope": float(c.get("slope", 1.0))}
            for c in config.get("candidates", [])
        ]
        self.sat_rate = float(config.get("sat_rate", 0.05))
        self.rain_multiplier = float(config.get("rain_multiplier", 2.0))
        self.raining = bool(config.get("raining", True))
        # Weather coupling (#18): explicit weather overrides raining.
        self.weather_active = config.get("weather_active")
        modifiers = config.get("weather_modifiers", {})
        if self.weather_active is not None:
            if self.weather_active not in modifiers:
                raise ValueError(
                    f"Unknown terrain weather {self.weather_active!r} "
                    f"(have {sorted(modifiers)})"
                )
            self.raining = bool(
                modifiers[self.weather_active].get("raining", self.raining)
            )
        self.threshold = float(config.get("threshold", 1.0))
        debris = tuple(config.get("debris_dir", [0, 1]))
        self.debris_dir = (int(debris[0]), int(debris[1]))
        self.debris_len = int(config.get("debris_len", 3))
        self.saturation = 0.0
        self.slip_step: int | None = None
        self.debris_cells: set[tuple[int, int]] = set()

    def step(self, current_step: int) -> set[tuple[int, int]]:
        """Advance; return debris cells if the slip triggers this step."""
        if self.slip_step is not None or not self.candidates:
            return set()
        rate = self.sat_rate * (self.rain_multiplier if self.raining else 1.0)
        self.saturation += rate
        if self.saturation < self.threshold:
            return set()
        best = max(self.candidates, key=lambda c: c["slope"])
        cells = set()
        x, y = best["x"], best["y"]
        for _ in range(self.debris_len):
            if 0 <= x < self.width and 0 <= y < self.height:
                cells.add((x, y))
            x += self.debris_dir[0]
            y += self.debris_dir[1]
        self.slip_step = current_step
        self.debris_cells = cells
        return set(cells)

    def danger_steps(self) -> dict[tuple[int, int], int]:
        if self.slip_step is None:
            return {}
        return {cell: self.slip_step for cell in self.debris_cells}

    def snapshot(self) -> dict:
        return {
            "slip_step": self.slip_step,
            "debris": sorted(f"{x},{y}" for x, y in self.debris_cells),
            "saturation": round(self.saturation, 4),
        }
