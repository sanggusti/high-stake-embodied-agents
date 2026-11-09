"""Seg-map to grid costs to rescuer routes.

Grounded concept cells become a cost grid (roads attract, hazards repel,
blockage overlaps repel strongly); A* plans a ground route from the drone
to each located victim. Pure functions: the caller's `project_concept`
closure owns the pixel-to-grid step (see `dmb/perception/grounding.py`).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

from dmb.perception.grounding import (
    grid_from_state,
    pose_from_state,
    project_mask,
    survey_camera,
)
from dmb.perception.sam import HAZARD_CONCEPTS, SegMap, rasterize_concept

COST_ROAD = 0.5
COST_OPEN = 1.0
COST_HAZARD = 10.0
COST_BLOCKED = 25.0

ROAD_CONCEPT = "road"

# Bridges without a metre scale (mock) fall back to this; route output
# using it is labeled assumed.
ASSUMED_CELL_M = 5.0


@dataclass(frozen=True)
class Route:
    goal: tuple[int, int]
    waypoints: list[tuple[int, int]]  # start..goal inclusive; [] if unreachable
    reachable: bool
    cost: float

    @property
    def steps(self) -> int:
        return max(0, len(self.waypoints) - 1)

    def distance_m(self, cell_m: float) -> float:
        return self.steps * cell_m


def cost_grid(
    width: int,
    height: int,
    road: set[tuple[int, int]],
    hazard: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[list[float]]:
    """Per-cell traversal cost; blocked wins over road wins over open."""
    grid = [[COST_OPEN] * width for _ in range(height)]
    for x, y in road:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = COST_ROAD
    for x, y in hazard:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = COST_HAZARD
    for x, y in blocked:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = COST_BLOCKED
    return grid


def _neighbors(x: int, y: int, width: int, height: int):
    if x > 0:
        yield (x - 1, y)
    if x + 1 < width:
        yield (x + 1, y)
    if y > 0:
        yield (x, y - 1)
    if y + 1 < height:
        yield (x, y + 1)


def plan_route(
    grid: list[list[float]], start: tuple[int, int], goal: tuple[int, int]
) -> Route:
    """A* over a cost grid (4-connected); unreachable goals stay honest."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    if not (
        0 <= start[0] < width and 0 <= start[1] < height
        and 0 <= goal[0] < width and 0 <= goal[1] < height
    ):
        return Route(goal, [], False, float("inf"))
    if start == goal:
        return Route(goal, [start], True, 0.0)
    min_cost = min(min(row) for row in grid)
    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return Route(goal, path, True, g_score[goal])
        if current in closed:
            continue
        closed.add(current)
        for nb in _neighbors(*current, width, height):
            tentative = g_score[current] + grid[nb[1]][nb[0]]
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                heuristic = (abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])) * min_cost
                heapq.heappush(open_heap, (tentative + heuristic, nb))
    return Route(goal, [], False, float("inf"))


def ground_concepts(
    project_concept: Callable[[str], set[tuple[int, int]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Ground road + hazard concepts; returns (road, hazard, blocked) cells."""
    road = project_concept(ROAD_CONCEPT) or set()
    hazard: set[tuple[int, int]] = set()
    for concept in HAZARD_CONCEPTS:
        hazard |= project_concept(concept) or set()
    return road, hazard, road & hazard


def plan_routes(
    grid: list[list[float]],
    start: tuple[int, int],
    goals: list[tuple[int, int]],
) -> list[Route]:
    return [plan_route(grid, start, goal) for goal in goals]


def render_route_map(
    seg: SegMap,
    state: dict,
    cell_m: float,
    assumed_geometry: bool = True,
    stride: int = 2,
    cam=None,
    pose=None,
) -> str:
    """Shared tool/solver composition: seg-map + state -> rescuer text.

    `cam`/`pose` override the assumed survey stance (survey frames pass
    their calibrated camera + pose with assumed_geometry=False).
    """
    if seg.frame_size is None:
        return "Route map unavailable: empty seg-map."
    cam = cam or survey_camera(*seg.frame_size)
    pose = pose or pose_from_state(state)
    spec = grid_from_state(state, cell_m)

    def project_concept(concept: str):
        result = seg.results.get(concept)
        if result is None:
            return set()
        cells: set[tuple[int, int]] = set()
        for record, raster in zip(result.records, result.rasters):
            if raster is None:
                continue
            cells |= project_mask(record, raster, cam, pose, spec, stride)
        return cells

    road, hazard, blocked = ground_concepts(project_concept)
    grid = cost_grid(spec.width, spec.height, road, hazard, blocked)
    start = (int(state["drone"]["x"]), int(state["drone"]["y"]))
    goals = [
        (int(v["x"]), int(v["y"]))
        for v in state.get("victims", []) if v.get("found")
    ]
    return format_route_map(
        seg, plan_routes(grid, start, goals), start, cell_m, assumed_geometry)


SEG_OVERLAY_COLORS = {
    "fire": (255, 60, 20),
    # Deep blue + strong alpha: smoke masks sit on white columns, so a
    # pale tint would wash out.
    "smoke": (70, 150, 255),
    "road": (255, 220, 0),
    "water": (40, 120, 255),
    "person": (60, 255, 60),
    "animal": (200, 80, 255),
    "building": (80, 220, 220),
    "vehicle": (255, 80, 200),
    "car": (255, 80, 200),
    "tent": (255, 160, 40),
}


def render_seg_overlay(png: bytes, seg: SegMap, grid: int = 160) -> bytes:
    """Final-segmentation footage: frame + translucent concept masks."""
    import io  # noqa: PLC0415

    from PIL import Image, ImageDraw  # noqa: PLC0415

    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if seg.frame_size:
        # Square cells over max(w,h), mirroring rasterize_concept.
        cell_px = max(w, h) / grid
        for concept, result in seg.results.items():
            pairs = [
                (r, ras)
                for r, ras in zip(result.records, result.rasters)
                if ras is not None
            ]
            if not pairs:
                continue
            color = SEG_OVERLAY_COLORS.get(concept, (255, 255, 255))
            for gx, gy in rasterize_concept(pairs, seg.frame_size, grid):
                x0, y0 = gx * cell_px, gy * cell_px
                x1, y1 = x0 + cell_px, y0 + cell_px
                draw.rectangle([x0, y0, x1, y1], fill=color + (160,))
    base = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    counts = ",".join(f"{c}:{len(r.records)}" for c, r in seg.results.items())
    legend = (
        f"severity={seg.severity.label}({seg.severity.index:.2f}) "
        f"counts={{{counts}}}"
    )
    strip = Image.new("RGB", (w, 22), (10, 10, 10))
    ImageDraw.Draw(strip).text((6, 4), legend, fill=(240, 240, 240))
    out = Image.new("RGB", (w, h + 22), (0, 0, 0))
    out.paste(base, (0, 0))
    out.paste(strip, (0, h))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def save_seg_artifacts(sample_id: str, frame_png: bytes, seg: SegMap):
    """Persist final frame + segmentation overlay; None without a run id."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    run_id = os.environ.get("DMB_RUN_ID", "").strip()
    if not run_id:
        return None
    runs_dir = os.environ.get("DMB_RUNS_DIR", "").strip() or "runs"
    outdir = Path(runs_dir) / run_id / "sim"
    outdir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_id)
    (outdir / f"{safe}_frame.png").write_bytes(frame_png)
    (outdir / f"{safe}_segmap.png").write_bytes(
        render_seg_overlay(frame_png, seg)
    )
    return outdir


def format_route_map(
    seg: SegMap,
    routes: list[Route],
    start: tuple[int, int],
    cell_m: float | None,
    assumed_geometry: bool,
) -> str:
    """Deterministic rescuer-facing route text (pure; tool-tested)."""
    geometry = "assumed survey geometry" if assumed_geometry else "calibrated geometry"
    blockages = (
        ",".join(f"{b.label}:{b.coverage:.2f}" for b in seg.blockages) or "none"
    )
    lines = [
        f"Route map from {start} ({geometry}): "
        f"severity={seg.severity.label}({seg.severity.index:.2f}) "
        f"blockages=[{blockages}]."
    ]
    if not routes:
        lines.append("No located victims yet; keep searching.")
        return "\n".join(lines)
    for route in routes:
        gx, gy = route.goal
        if not route.reachable:
            lines.append(f"Victim ({gx},{gy}): UNREACHABLE from {start}.")
            continue
        dist = f"{route.distance_m(cell_m):.1f} m, " if cell_m else ""
        path = " -> ".join(f"({x},{y})" for x, y in route.waypoints)
        lines.append(f"Victim ({gx},{gy}): {route.steps} steps, {dist}path: {path}.")
    return "\n".join(lines)
