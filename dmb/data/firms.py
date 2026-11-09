"""NASA FIRMS active-fire data -> scenario truth (sim-to-real ground).

Parses FIRMS archive CSVs (VIIRS C2 columns), projects hotspots around an
event anchor into grid cells, and splits day-0 ignitions from the day-1..3
spread reference. Comparison happens at satellite-block scale (75 cells =
375 m at 5 m/cell): sim output is downsampled to the sensor, never the
reverse.

Caveats (methodology, not bugs): first observed day in-window stands in
for ignition day (cloud gaps exist); hotspot FRP weights intensity but
smoldering peat reads low FRP while still burning.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

CELL_M = 5.0
VIIRS_BLOCK_CELLS = 75  # 375 m sensor pixel at 5 m/cell


@dataclass(frozen=True)
class Hotspot:
    lat: float
    lon: float
    date: str
    frp: float


@dataclass(frozen=True)
class FirmsEvent:
    lat: float  # anchor (local origin)
    lon: float
    t0: str  # ignition day YYYY-MM-DD
    grid: int = 120  # square grid; anchor maps to center cell


def load_hotspots(path: str) -> list[Hotspot]:
    """Parse a FIRMS archive CSV; malformed rows are skipped, never fatal."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                date = str(row["acq_date"])
                frp = float(row.get("frp") or 0.0)
            except (ValueError, TypeError, KeyError):
                continue
            if not date or len(date) != 10:
                continue
            out.append(Hotspot(lat, lon, date, frp))
    return out


def project(event: FirmsEvent, lat: float, lon: float) -> tuple[int, int] | None:
    """Hotspot -> grid cell (local equirectangular, anchor at center)."""
    east = (lon - event.lon) * 111320.0 * math.cos(math.radians(event.lat))
    north = (lat - event.lat) * 110540.0
    half = event.grid * CELL_M / 2.0
    if not (-half <= east < half and -half <= north < half):
        return None
    return (
        min(event.grid - 1, int((east + half) // CELL_M)),
        min(event.grid - 1, int((north + half) // CELL_M)),
    )


def split_event(
    hotspots: list[Hotspot], event: FirmsEvent, ref_days: int = 3
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    """Day-0 ignition cells, day-1..N NEW-block reference cells, FRP stats."""
    from datetime import date as _date

    t0 = _date.fromisoformat(event.t0)
    ref_dates = {
        str(_date.fromordinal(t0.toordinal() + i)) for i in range(1, ref_days + 1)
    }
    ignitions: dict[tuple[int, int], float] = {}
    ref_cells: set[tuple[int, int]] = set()
    for hotspot in hotspots:
        cell = project(event, hotspot.lat, hotspot.lon)
        if cell is None:
            continue
        if hotspot.date == event.t0:
            ignitions[cell] = max(ignitions.get(cell, 0.0), hotspot.frp)
        elif hotspot.date in ref_dates:
            ref_cells.add(cell)
    ign_blocks = {
        (x // VIIRS_BLOCK_CELLS, y // VIIRS_BLOCK_CELLS) for x, y in ignitions
    }
    ref_blocks = sorted(
        {
            (x // VIIRS_BLOCK_CELLS, y // VIIRS_BLOCK_CELLS)
            for x, y in ref_cells
        }
        - ign_blocks
    )
    frps = list(ignitions.values())
    stats = {
        "n_ignition_cells": len(ignitions),
        "n_ref_blocks": len(ref_blocks),
        "frp_mean": round(sum(frps) / len(frps), 1) if frps else 0.0,
        "frp_max": round(max(frps), 1) if frps else 0.0,
    }
    return sorted(ignitions), ref_blocks, stats
