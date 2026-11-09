"""Flood water: advancing surge front, currents, victim drift (T0).

A front advances from an origin edge at a fixed rate; cells behind it are
flooded (hazardous). Mobile victims in flooded cells drift along the
current every few steps. Deterministic: analytic front position, no RNG.
"""

from __future__ import annotations


class WaterModel:
    def __init__(self, config: dict, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.origin = str(config.get("origin", "south"))
        self.rate_num = int(config.get("rate_num", 1))
        self.rate_den = int(config.get("rate_den", 4))
        current = tuple(config.get("current", [0, -1]))
        self.current = (int(current[0]), int(current[1]))
        self.drift_every = int(config.get("drift_every", 3))
        self._flooded: set[tuple[int, int]] = set()

    def front_position(self, step: int) -> int:
        return (self.rate_num * step) // max(1, self.rate_den)

    def flooded_at(self, step: int) -> set[tuple[int, int]]:
        front = self.front_position(step)
        cells = set()
        for x in range(self.width):
            for y in range(self.height):
                if self.origin == "south" and y >= self.height - front:
                    cells.add((x, y))
                elif self.origin == "north" and y < front:
                    cells.add((x, y))
                elif self.origin == "west" and x < front:
                    cells.add((x, y))
                elif self.origin == "east" and x >= self.width - front:
                    cells.add((x, y))
        return cells

    def flood_step(self, cell: tuple[int, int]) -> int | None:
        """First step a cell floods (analytic inverse), or None if never."""
        x, y = cell
        if self.origin == "south":
            depth = self.height - y
        elif self.origin == "north":
            depth = y + 1
        elif self.origin == "west":
            depth = x + 1
        else:
            depth = self.width - x
        if self.rate_num <= 0:
            return None
        # front(step) >= depth  <=>  step >= ceil(depth * den / num)
        from math import ceil

        return ceil(depth * self.rate_den / self.rate_num)

    def step(self, current_step: int) -> set[tuple[int, int]]:
        """Advance; return newly flooded cells."""
        now = self.flooded_at(current_step)
        new = now - self._flooded
        self._flooded = now
        return new

    def drift(self, victim, current_step: int) -> bool:
        """Move a mobile flooded victim one cell along the current."""
        if victim.mobility != "mobile":
            return False
        if (victim.x, victim.y) not in self._flooded:
            return False
        if current_step % max(1, self.drift_every) != 0:
            return False
        victim.x = max(0, min(self.width - 1, victim.x + self.current[0]))
        victim.y = max(0, min(self.height - 1, victim.y + self.current[1]))
        return True

    def danger_steps(self) -> dict[tuple[int, int], int]:
        out: dict[tuple[int, int], int] = {}
        for x in range(self.width):
            for y in range(self.height):
                step = self.flood_step((x, y))
                if step is not None:
                    out[(x, y)] = step
        return out

    def snapshot(self) -> dict:
        return {"flooded": sorted(f"{x},{y}" for x, y in self._flooded)}
