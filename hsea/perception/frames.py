"""Frame pipeline: truth-annotated observations for the vision channel.

Two sources, one interface:
- rendered map frames (always available): top-down PNG drawn from sim state.
- native Gazebo frames (SITL workers): camera pixels + telemetry overlay.

Observation honesty rule: frames show only what the agent could observe —
visible dynamics (burning/flooded/debris cells), FOUND victims, drone pose,
telemetry. Unfound victims and invisible static zones are never drawn.
"""

from __future__ import annotations

import base64
import io
import subprocess

from PIL import Image, ImageDraw

CELL_PX = 48
FOUND_COLOR = (0, 255, 0)
DRONE_COLOR = (255, 255, 0)
FIRE_COLOR = (255, 128, 0)
WATER_COLOR = (0, 128, 255)
DEBRIS_COLOR = (160, 82, 45)
BG_COLOR = (24, 28, 36)
GRID_COLOR = (60, 66, 80)


def render_map_frame(state: dict, grid: dict) -> bytes:
    """Top-down annotated PNG from episode state. No leakage by construction."""
    width, height = int(grid.get("width", 10)), int(grid.get("height", 10))
    img = Image.new("RGB", (width * CELL_PX, height * CELL_PX + 28), BG_COLOR)
    draw = ImageDraw.Draw(img)
    steps = int(state.get("steps", 0))
    danger = state.get("cell_danger", {})
    for x in range(width):
        for y in range(height):
            x0, y0 = x * CELL_PX, y * CELL_PX
            draw.rectangle([x0, y0, x0 + CELL_PX - 1, y0 + CELL_PX - 1], outline=GRID_COLOR)
            key = f"{x},{y}"
            if key in danger and danger[key] <= steps:
                # Visible dynamics only: danger color varies by scenario via
                # the kind recorded in state (default fire-orange).
                color = {"fire": FIRE_COLOR, "water": WATER_COLOR, "terrain": DEBRIS_COLOR}.get(
                    state.get("danger_kind", "fire"), FIRE_COLOR
                )
                draw.rectangle(
                    [x0 + 2, y0 + 2, x0 + CELL_PX - 3, y0 + CELL_PX - 3], fill=color
                )
    for victim in state.get("victims", []):
        if not victim.get("found"):
            continue
        cx, cy = victim["x"] * CELL_PX + CELL_PX // 2, victim["y"] * CELL_PX + CELL_PX // 2
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=FOUND_COLOR)
        draw.text((cx - 4, cy - 7), victim["category"][0].upper(), fill=(0, 0, 0))
    drone = state.get("drone", {})
    dx, dy = int(drone.get("x", 0)) * CELL_PX + CELL_PX // 2, int(drone.get("y", 0)) * CELL_PX + CELL_PX // 2
    draw.polygon([(dx, dy - 12), (dx - 9, dy + 8), (dx + 9, dy + 8)], fill=DRONE_COLOR)
    caption = (
        f"{state.get('scenario', '?')} step={steps} phase={state.get('phase', '?')} "
        f"pos=({drone.get('x', '?')},{drone.get('y', '?')}) batt={drone.get('battery_pct', '?')}%"
    )
    draw.text((6, height * CELL_PX + 6), caption, fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SENSOR_DOT = (220, 220, 220)
SENSOR_SIZES = {"child": 5, "human": 8, "animal": 6}


def render_sensor_frame(state: dict, grid: dict, radius: float) -> bytes:
    """Raw sensor view for the edge VLM: unclassified contacts in range.

    Every victim within `radius` of the drone is drawn as a size-coded dot
    (size hints category, like appearance would); nothing is labeled. The
    VLM must localize and classify from pixels — that IS the perception
    work. Victims out of range are not drawn (sensor horizon).
    """
    import math

    width, height = int(grid.get("width", 10)), int(grid.get("height", 10))
    drone = state.get("drone", {})
    dx, dy = int(drone.get("x", 0)), int(drone.get("y", 0))
    img = Image.new("RGB", (width * CELL_PX, height * CELL_PX), (8, 10, 14))
    draw = ImageDraw.Draw(img)
    for victim in state.get("victims", []):
        if math.hypot(victim["x"] - dx, victim["y"] - dy) > radius:
            continue
        cx, cy = victim["x"] * CELL_PX + CELL_PX // 2, victim["y"] * CELL_PX + CELL_PX // 2
        r = SENSOR_SIZES.get(victim["category"], 6)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SENSOR_DOT)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_telemetry(png_bytes: bytes, lines: list[str]) -> bytes:
    """Stamp telemetry text rows onto a native frame (top-left, legible)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        w = int(draw.textlength(line)) + 12
        draw.rectangle([6, y - 2, 6 + w, y + 14], fill=(0, 0, 0))
        draw.text((10, y), line, fill=(255, 255, 0))
        y += 20
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


class GazeboFrameCapture:
    """Native camera frames from a running SITL worker container."""

    def __init__(self, container_id: str, partition: str, timeout_s: float = 45.0) -> None:
        self.container_id = container_id
        self.partition = partition
        self.timeout_s = timeout_s

    def capture(self, ref: str) -> bytes | None:
        """Grab one frame; return PNG bytes or None on any failure."""
        remote = f"/tmp/frame-{ref}.png"
        grab = subprocess.run(
            ["docker", "exec", "-e", f"GZ_PARTITION={self.partition}",
             self.container_id, "/home/pilot/grab_frame.py", remote],
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        if grab.returncode != 0 or not grab.stdout.startswith("OK "):
            return None
        cp = subprocess.run(
            ["docker", "cp", f"{self.container_id}:{remote}", "-"],
            capture_output=True, timeout=self.timeout_s,
        )
        if cp.returncode != 0 or not cp.stdout:
            return None
        return _extract_from_tar(cp.stdout)

    def close(self) -> None:
        return None


def _extract_from_tar(tar_bytes: bytes) -> bytes | None:
    """`docker cp -` streams a tar; pull the single file payload out."""
    import tarfile

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            if not members:
                return None
            extracted = tar.extractfile(members[0])
            return extracted.read() if extracted else None
    except (tarfile.TarError, EOFError, OSError):
        return None
