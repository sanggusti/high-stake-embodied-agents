"""Pixel-to-grid grounding for segmentation masks.

Pinhole camera, flat-ground intersection: a mask pixel becomes a ray from
the drone pose, meets the z=0 plane, and lands in a grid cell. World frame
matches the sim bridges: X=east, Y=north, Z=up, cell (x, y) covering
``[x*cell_m, (x+1)*cell_m)`` east by ``[y*cell_m, (y+1)*cell_m)`` north.

No calibrated camera exists yet (no FOV/orientation is documented for the
Gazebo gimbal or the Colosseum front camera), so callers pass explicit
parameters. `survey_camera` is the documented ASSUMED survey stance (gimbal
commanded to nadir for route mapping); backend calibration is later work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera with square pixels and a centered principal point."""

    width: int
    height: int
    hfov_deg: float  # horizontal field of view, (0, 180)
    pitch_deg: float  # optical axis below horizon; 0 = level, 90 = nadir
    yaw_deg: float = 0.0  # clockwise from north

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image size must be positive")
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError(f"hfov {self.hfov_deg} outside (0, 180)")
        if not 0.0 <= self.pitch_deg <= 90.0:
            raise ValueError(f"pitch {self.pitch_deg} outside [0, 90]")

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


@dataclass(frozen=True)
class DronePose:
    x: int  # grid cell
    y: int
    altitude_m: float


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    cell_m: float


def survey_camera(width: int, height: int, hfov_deg: float = 90.0) -> CameraModel:
    """ASSUMED route-mapping survey stance: gimbal to nadir, yaw north.

    Placeholder until backend camera calibration lands; outputs derived
    from it must be labeled assumed, never cited as surveyed positions.
    """
    return CameraModel(width, height, hfov_deg, 90.0, 0.0)


def pose_from_state(state: dict) -> DronePose:
    drone = state.get("drone", {})
    return DronePose(
        int(drone.get("x", 0)), int(drone.get("y", 0)),
        float(drone.get("altitude_m", 0.0)),
    )


def grid_from_state(state: dict, cell_m: float) -> GridSpec:
    grid = state.get("grid", {})
    return GridSpec(int(grid.get("width", 10)), int(grid.get("height", 10)), cell_m)


def _basis(cam: CameraModel) -> tuple[tuple[float, float, float], ...]:
    """Camera (right, down, forward) unit vectors in world coordinates."""
    pitch = math.radians(cam.pitch_deg)
    yaw = math.radians(cam.yaw_deg % 360.0)
    cp, sp = math.cos(pitch), math.sin(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    fwd = (cp * sy, cp * cy, -sp)
    if cp < 1e-9:  # nadir: f x up degenerates, use yaw directly
        return ((cy, -sy, 0.0), (-sy, -cy, 0.0), fwd)
    rx, ry = fwd[1], -fwd[0]
    norm = math.hypot(rx, ry)
    right = (rx / norm, ry / norm, 0.0)
    down = (
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0],
    )
    return (right, down, fwd)


def project_pixel(
    u: float, v: float, cam: CameraModel, pose: DronePose, grid: GridSpec
) -> tuple[int, int] | None:
    """Map a pixel to a grid cell, or None (sky ray, grounded, off-grid)."""
    if pose.altitude_m <= 0 or grid.cell_m <= 0:
        return None
    right, down, fwd = _basis(cam)
    rx = (u - cam.cx) / cam.fx
    ry = (v - cam.cy) / cam.fy
    dx = rx * right[0] + ry * down[0] + fwd[0]
    dy = rx * right[1] + ry * down[1] + fwd[1]
    dz = rx * right[2] + ry * down[2] + fwd[2]
    if dz >= 0:
        return None
    t = -pose.altitude_m / dz
    east = pose.x * grid.cell_m + t * dx
    north = pose.y * grid.cell_m + t * dy
    x = math.floor(east / grid.cell_m)
    y = math.floor(north / grid.cell_m)
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return None
    return (x, y)


def grid_to_pixel(
    cell: tuple[int, int], cam: CameraModel, pose: DronePose, grid: GridSpec
) -> tuple[float, float] | None:
    """Inverse map for nadir cameras: grid cell center -> pixel.

    Exact for pitch=90 at any yaw (non-nadir raises); used by the
    perception oracle to place perfect boxes and by the SAM proposer
    to seed point prompts. Returns None when the cell falls outside
    the frame.
    """
    if cam.pitch_deg != 90.0:
        raise ValueError("grid_to_pixel needs the nadir survey stance")
    if pose.altitude_m <= 0 or grid.cell_m <= 0:
        return None
    yaw = math.radians(cam.yaw_deg % 360.0)
    sy, cy = math.sin(yaw), math.cos(yaw)
    east = (cell[0] + 0.5) * grid.cell_m
    north = (cell[1] + 0.5) * grid.cell_m
    dx = (east - pose.x * grid.cell_m) / pose.altitude_m
    dy = (north - pose.y * grid.cell_m) / pose.altitude_m
    # Inverse of the nadir ray: rx = cy*dx - sy*dy, ry = -sy*dx - cy*dy.
    u = cam.cx + cam.fx * (cy * dx - sy * dy)
    v = cam.cy + cam.fy * (-sy * dx - cy * dy)
    if not (0 <= u < cam.width and 0 <= v < cam.height):
        return None
    return (u, v)


def project_mask(
    record,  # SAMRecord (duck-typed to keep this module dependency-free)
    raster: bytes,
    cam: CameraModel,
    pose: DronePose,
    grid: GridSpec,
    stride: int = 1,
) -> set[tuple[int, int]]:
    """Ground every set raster pixel (subsampled by `stride`) to grid cells."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1 (got {stride})")
    box = record.box
    bw = box.x2 + 1 - box.x1
    bh = box.y2 + 1 - box.y1
    mh, mw = record.mask.height, record.mask.width
    cells: set[tuple[int, int]] = set()
    if mh <= 0 or mw <= 0 or len(raster) != mh * mw:
        return cells
    for r in range(0, mh, stride):
        v = box.y1 + (r + 0.5) * bh / mh
        base = r * mw
        for c in range(0, mw, stride):
            if not raster[base + c]:
                continue
            u = box.x1 + (c + 0.5) * bw / mw
            cell = project_pixel(u, v, cam, pose, grid)
            if cell is not None:
                cells.add(cell)
    return cells
