"""Pre-rendered survey frames: the sim-to-real frame bridge for SAM.

SAM 3.1 returns zero matches on MockBridge 2D map renders (prior
finding), but prompted delineation works on the Gazebo survey renders
(firebed/smoke masks verified 2026-11-04). This module serves those
pre-rendered frames -- with their exact survey camera geometry -- to
SAM scans on firms scenarios, so the SAM-vs-stub comparison measures
segmentation on real 3D renders instead of SAM-blind map art.

Layout under the frames roots (default docs/firms-sim2real/renders/
then docs/quake-slide-sim2real/renders/, override DMB_RENDER_FRAMES_DIR):
  survey-<scenario>-<seed>-nadir.png   the Map POV frame
  survey-<scenario>-<seed>-nadir.json  camera sidecar (see below)

Sidecar schema: width, height, hfov_deg, pitch_deg, yaw_deg,
pose_cell [x, y], altitude_m. The Kalimantan seed-7 nadir geometry
was verified empirically (<15 px on both firebeds).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dmb.perception.grounding import CameraModel, DronePose


@dataclass(frozen=True)
class SurveyFrame:
    png: bytes
    cam: CameraModel
    pose: DronePose


def frames_roots() -> list[Path]:
    override = os.environ.get("DMB_RENDER_FRAMES_DIR", "").strip()
    if override:
        return [Path(override)]
    docs = Path(__file__).resolve().parents[2] / "docs"
    return [
        docs / "firms-sim2real" / "renders",
        docs / "quake-slide-sim2real" / "renders",
        docs / "flood-sim2real" / "renders",
    ]


def frames_root() -> Path:
    return frames_roots()[0]


def survey_frame(scenario: str, seed: int) -> SurveyFrame | None:
    """Load the survey frame + camera for (scenario, seed), if present."""
    for root in frames_roots():
        base = root / f"survey-{scenario}-{seed}-nadir"
        png_path = base.with_suffix(".png")
        js_path = base.with_suffix(".json")
        if png_path.exists() and js_path.exists():
            break
    else:
        return None
    spec = json.loads(js_path.read_text())
    cam = CameraModel(
        int(spec["width"]),
        int(spec["height"]),
        float(spec["hfov_deg"]),
        float(spec["pitch_deg"]),
        float(spec.get("yaw_deg", 0.0)),
    )
    pose = DronePose(
        int(spec["pose_cell"][0]),
        int(spec["pose_cell"][1]),
        float(spec["altitude_m"]),
    )
    return SurveyFrame(png_path.read_bytes(), cam, pose)
