"""Perception oracle: perfect seg-maps from sim truth (ceiling evals).

The oracle answers "what would perfect perception achieve" by projecting
sim ground truth (activated danger cells, victim cells) into mask records
through the survey camera. The route planner, grounding, cost grid, and
scorers run for real; only the detector is perfect. No network, fully
deterministic.

Honesty notes: records exist only for in-frame cells (off-frame truth is
skipped, like any sensor horizon), and terrain danger emits no hazard
concept — matching production, where rubble/debris are unsegmentable.
"""

from __future__ import annotations

from typing import Callable

from dmb.perception.grounding import (
    grid_from_state,
    grid_to_pixel,
    pose_from_state,
    survey_camera,
)
from dmb.perception.sam import (
    SAMBox,
    SAMMask,
    SAMRecord,
    SegConceptResult,
    SegMap,
    apply_overlap_rules,
    rasterize_concept,
    severity_of,
)

# Sim dynamics kind -> segmentable hazard concept (terrain has none).
ORACLE_HAZARD_CONCEPT = {"fire": "smoke", "water": "water"}

ORACLE_GRID = 64


def frame_size(png: bytes) -> tuple[int, int]:
    from PIL import Image  # noqa: PLC0415
    import io  # noqa: PLC0415

    with Image.open(io.BytesIO(png)) as img:
        return img.size


def _record_for_cell(
    object_id: int, cell: tuple[int, int], u: float, v: float,
    frame_wh: tuple[int, int],
) -> tuple[SAMRecord, bytes]:
    fw, fh = frame_wh
    x1 = max(0, min(fw - 1, int(u) - 2))
    y1 = max(0, min(fh - 1, int(v) - 2))
    x2 = max(x1, min(fw - 1, int(u) + 2))
    y2 = max(y1, min(fh - 1, int(v) + 2))
    record = SAMRecord(
        object_id=object_id,
        box=SAMBox(x1, y1, x2, y2, fw, fh, 1.0),
        mask=SAMMask(1, 1, "one_bit", "!1"),
    )
    return record, bytes([1])


def oracle_seg_provider(
    bridge, cell_m: float = 5.0  # MockBridge (duck-typed)
) -> Callable[[bytes], SegMap]:
    """Build a `seg_provider` projecting this bridge's truth into a SegMap."""

    def provide(png: bytes) -> SegMap:
        state = bridge.get_state()
        frame_wh = frame_size(png)
        cam = survey_camera(*frame_wh)
        pose = pose_from_state(state)
        spec = grid_from_state(state, cell_m)
        kinds = set((state.get("dynamics", {}) or {}).keys())
        hazard_concepts = {
            ORACLE_HAZARD_CONCEPT[k] for k in kinds if k in ORACLE_HAZARD_CONCEPT
        }

        def project_cells(cells: list[tuple[int, int]]):
            records, rasters = [], []
            for cell in cells:
                pixel = grid_to_pixel(cell, cam, pose, spec)
                if pixel is None:
                    continue
                record, raster = _record_for_cell(
                    len(records), cell, pixel[0], pixel[1], frame_wh)
                records.append(record)
                rasters.append(raster)
            return records, rasters

        results: dict[str, SegConceptResult] = {}
        for concept in ("road", "building", "person", "vehicle", "car",
                        "animal", "water", "smoke"):
            results[concept] = SegConceptResult(concept, [], [])
        from dmb.sim.bridge import danger_cells_from_state  # noqa: PLC0415

        danger_cells = sorted(danger_cells_from_state(state))
        for concept in hazard_concepts:
            records, rasters = project_cells(sorted(set(danger_cells)))
            results[concept] = SegConceptResult(concept, records, rasters)
        persons, animals = [], []
        for victim in state.get("victims", []) or []:
            try:
                cell = (int(victim["x"]), int(victim["y"]))
            except (ValueError, TypeError, KeyError):
                continue
            (animals if victim.get("category") == "animal" else persons).append(cell)
        records, rasters = project_cells(sorted(set(persons)))
        results["person"] = SegConceptResult("person", records, rasters)
        records, rasters = project_cells(sorted(set(animals)))
        results["animal"] = SegConceptResult("animal", records, rasters)

        seg = SegMap(results=results, frame_size=frame_wh)
        cells = {
            concept: rasterize_concept(
                list(zip(result.records, result.rasters)), frame_wh, ORACLE_GRID)
            for concept, result in results.items()
        }
        seg.blockages = apply_overlap_rules(cells)
        hazard_union: set[tuple[int, int]] = set().union(
            *(cells.get(c, set()) for c in ("water", "smoke")))
        seg.severity = severity_of(hazard_union, ORACLE_GRID * ORACLE_GRID)
        return seg

    return provide
