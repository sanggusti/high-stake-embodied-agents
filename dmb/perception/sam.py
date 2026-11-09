"""SAM 3.1 segmentation maps for route mapping and severity.

Drone footage goes to Meta's hosted `sam-3.1` model (Responses API); one
request per concept returns boxes + masks. This module turns those into a
`SegMap`: per-concept records, derived route blockages (mask overlaps such
as road ∩ water = flooded road), and a severity index for the recon
advisor. `SAMEdgePerception` exposes it through the standard edge
interface with the same triggers/budget as the rule stub.

Two evidence-backed limits (see docs/sam-segmap.md):
- SAM segments photorealistic native frames only. MockBridge map renders
  return zero matches, so this layer is effective on gazebo-sitl /
  Colosseum captures and degrades to empty reports on mock.
- Amorphous damage nouns ("rubble", "debris", "pile", "collapsed
  building") return zero as concepts. Rubble is delineated via
  point/box prompts (`rubble_point_prompt`) seeded from road-mask gaps or
  operator clicks, which segments precisely (conf 0.906 observed).

Live use needs `DMB_SAM_API_KEY` (or the shared `META_AI_API_KEY`).
Unit tests inject a stub segmenter — no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from meta_sam_parser import (
    FramePrompt,
    PromptObject,
    PromptPoint,
    SegmentationMask,
    build_object_prompt,
    decode_mask_to_raster,
)

from dmb.perception.edge import Detection, StubEdgePerception
from dmb.perception.grounding import GridSpec, grid_to_pixel

SAM_DEFAULT_BASE_URL = "https://api.meta.ai/v1"
SAM_DEFAULT_MODEL = "sam-3.1"

# Concepts requested on every frame (all proven on aerial footage except
# "animal", which is oracle-covered but unbaselined on real footage).
BASE_CONCEPTS = ("road", "building", "person", "vehicle", "car", "animal")
# Concepts whose frame coverage feeds the severity index ("fire" = sim-seeded
# point-prompt firebed masks on survey frames; no scenario requests it openly;
# "rubble"/"debris" likewise from hazard-zone point prompts on quake/slide).
HAZARD_CONCEPTS = ("water", "smoke", "fire", "rubble", "debris")

# Per-scenario extras keyed by base scenario name; `_large` variants resolve
# to their base. Only concepts with observed matches are listed.
SCENARIO_CONCEPTS = {
    "forest_fire": ("smoke",),
    "tsunami": ("water",),
    "flash_flood": ("water",),
    "flood": ("water",),
    "earthquake": ("tent",),  # relief camps; rubble via point prompts
    "landslide": (),
}

# (passable concept, hazard concept, derived blockage label). State phrases
# ("flooded road") are unreliable as prompts, so blockage is derived here
# from plain-concept mask overlap.
OVERLAP_RULES = (
    ("road", "water", "flooded_road"),
    ("road", "smoke", "smoke_blocked_road"),
)

# Severity cutoffs on hazard frame fraction (observed: floodwater 0.36 and
# smoke 0.23 read severe; minor water 0.04 reads minor).
SEVERITY_MODERATE_AT = 0.05
SEVERITY_SEVERE_AT = 0.20

# SAM concepts whose detections ground to victims: concept -> categories.
VICTIM_CONCEPTS = (("person", ("human", "child")), ("animal", ("animal",)))


def concepts_for_scenario(name: str) -> tuple[str, ...]:
    """Concepts to segment for a scenario (base + hazard extras)."""
    base = name
    if base not in SCENARIO_CONCEPTS:
        base = base.removesuffix("_large").removesuffix("_hard")
    if base not in SCENARIO_CONCEPTS and name.startswith("firms_peat"):
        base = "forest_fire"  # FIRMS peat scenarios share the fire extras
    if base not in SCENARIO_CONCEPTS:
        raise ValueError(
            f"Unknown scenario {name!r} (have {sorted(SCENARIO_CONCEPTS)})"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for concept in BASE_CONCEPTS + SCENARIO_CONCEPTS[base]:
        if concept not in seen:
            seen.add(concept)
            ordered.append(concept)
    return tuple(ordered)


@dataclass(frozen=True)
class SAMBox:
    """One box in source pixels (`x2`/`y2` inclusive, per the SAM grammar)."""

    x1: int
    y1: int
    x2: int
    y2: int
    w: int
    h: int
    confidence: float | None = None

    @property
    def right(self) -> int:
        return self.x2 + 1  # half-open, for raster mapping

    @property
    def bottom(self) -> int:
        return self.y2 + 1

    @property
    def area(self) -> int:
        return (self.x2 + 1 - self.x1) * (self.y2 + 1 - self.y1)


@dataclass(frozen=True)
class SAMMask:
    """One mask: `height`×`width` raster; payload keeps its `!`/`~` marker."""

    height: int
    width: int
    encoding: str  # "lossless" | "one_bit"
    payload: str


@dataclass(frozen=True)
class SAMRecord:
    object_id: int
    box: SAMBox
    mask: SAMMask

    @property
    def confidence(self) -> float | None:
        return self.box.confidence


_FRAME_RE = re.compile(r"<(\d+)f>")
# The payload alphabet never contains `|`, so `[^|]*` ends each token exactly.
_RECORD_RE = re.compile(r"(\d+)(<\|box;[^|]*\|>)(<\|mask;[^|]*\|>)")
_MASK_DATA_RE = re.compile(r"^(\d+),(\d+),([!~].+)$")


def _fields(token: str) -> dict[str, str]:
    inner = token[len("<|box;") if token.startswith("<|box;") else len("<|mask;"):]
    inner = inner[: -len("|>")]
    out: dict[str, str] = {}
    for part in inner.split(";"):
        if "=" in part:
            key, _, value = part.partition("=")
            out[key] = value
    return out


def _parse_box(token: str) -> SAMBox | None:
    try:
        f = _fields(token)
        x1, y1, x2, y2 = int(f["x1"]), int(f["y1"]), int(f["x2"]), int(f["y2"])
        w, h = int(f["w"]), int(f["h"])
        conf = float(f["c"]) if "c" in f else None
    except (KeyError, ValueError):
        return None
    if w <= 0 or h <= 0 or not (0 <= x1 <= x2 < w) or not (0 <= y1 <= y2 < h):
        return None
    return SAMBox(x1, y1, x2, y2, w, h, conf)


def _parse_mask(token: str) -> SAMMask | None:
    try:
        f = _fields(token)
        int(f["x"])
        int(f["y"])
        match = _MASK_DATA_RE.match(f["data"])
        if match is None:
            return None
        height, width, payload = int(match.group(1)), int(match.group(2)), match.group(3)
    except (KeyError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return SAMMask(height, width, "one_bit" if payload.startswith("!") else "lossless", payload)


def parse_sam_output(text: str) -> tuple[int | None, list[SAMRecord]]:
    """Parse one image output line; malformed records are dropped, never crash."""
    stripped = text.strip()
    if not stripped:
        return None, []
    frame_match = re.match(r"<(\d+)f>", stripped)
    if frame_match is None:
        return None, []
    records: list[SAMRecord] = []
    for match in _RECORD_RE.finditer(stripped):
        box = _parse_box(match.group(2))
        mask = _parse_mask(match.group(3))
        if box is None or mask is None:
            continue
        records.append(SAMRecord(int(match.group(1)), box, mask))
    return int(frame_match.group(1)), records


def parse_sam_video_output(text: str) -> dict[int, list[SAMRecord]]:
    """Parse multi-line video output, keyed by explicit frame marker."""
    out: dict[int, list[SAMRecord]] = {}
    for line in text.splitlines():
        frame, records = parse_sam_output(line)
        if frame is not None:
            out[frame] = records
    return out


def decode_record_mask(record: SAMRecord) -> bytes:
    """Decode a record mask to a row-major 0/1 raster (`height`×`width`)."""
    mask = SegmentationMask(
        encoding=record.mask.encoding,  # type: ignore[arg-type]
        payload=record.mask.payload,
        width=record.mask.width,
        height=record.mask.height,
    )
    return decode_mask_to_raster(mask)


def mask_fill(raster: bytes) -> float:
    """Fraction of raster pixels set."""
    return sum(raster) / len(raster) if raster else 0.0


def record_frame_fraction(record: SAMRecord, raster: bytes) -> float:
    """Fraction of the source frame covered by this record's mask."""
    frame_area = record.box.w * record.box.h
    if frame_area <= 0:
        return 0.0
    return mask_fill(raster) * record.box.area / frame_area


def rasterize_concept(
    pairs: list[tuple[SAMRecord, bytes]],
    frame_wh: tuple[int, int],
    grid: int = 64,
) -> set[tuple[int, int]]:
    """Union of concept masks as coarse grid cells over the frame.

    Each set raster pixel maps through its record's box into a `grid`×`grid`
    frame grid. Pure Python; grids stay small (default 64).
    """
    fw, fh = frame_wh
    cells: set[tuple[int, int]] = set()
    if fw <= 0 or fh <= 0 or grid <= 0:
        return cells
    cell = max(fw, fh) / grid
    for record, raster in pairs:
        box = record.box
        bw = box.x2 + 1 - box.x1
        bh = box.y2 + 1 - box.y1
        mh, mw = record.mask.height, record.mask.width
        if mh <= 0 or mw <= 0 or len(raster) != mh * mw:
            continue
        for r in range(mh):
            gy = min(grid - 1, max(0, int((box.y1 + (r + 0.5) * bh / mh) / cell)))
            base = r * mw
            for c in range(mw):
                if raster[base + c]:
                    gx = min(grid - 1, max(0, int((box.x1 + (c + 0.5) * bw / mw) / cell)))
                    cells.add((gx, gy))
    return cells


@dataclass(frozen=True)
class DerivedBlockage:
    label: str
    covers: str  # passable concept, e.g. "road"
    covered_by: str  # hazard concept, e.g. "water"
    coverage: float  # fraction of `covers` cells overlapped
    cells: int  # overlapped coarse cells


def apply_overlap_rules(
    cells_by_concept: dict[str, set[tuple[int, int]]],
    rules: tuple[tuple[str, str, str], ...] = OVERLAP_RULES,
    min_coverage: float = 0.02,
) -> list[DerivedBlockage]:
    """Derive route blockages from concept-masks overlaps."""
    out: list[DerivedBlockage] = []
    for covers, covered_by, label in rules:
        base = cells_by_concept.get(covers) or set()
        hazard = cells_by_concept.get(covered_by) or set()
        if not base or not hazard:
            continue
        overlap = base & hazard
        coverage = len(overlap) / len(base)
        if coverage < min_coverage:
            continue
        out.append(DerivedBlockage(label, covers, covered_by, coverage, len(overlap)))
    return out


@dataclass(frozen=True)
class SeverityScore:
    index: float  # hazard frame fraction in [0, 1]
    label: str  # "minor" | "moderate" | "severe"


def severity_of(hazard_cells: set[tuple[int, int]], grid_total: int) -> SeverityScore:
    index = len(hazard_cells) / grid_total if grid_total > 0 else 0.0
    if index >= SEVERITY_SEVERE_AT:
        label = "severe"
    elif index >= SEVERITY_MODERATE_AT:
        label = "moderate"
    else:
        label = "minor"
    return SeverityScore(index, label)


@dataclass
class SegConceptResult:
    concept: str
    records: list[SAMRecord] = field(default_factory=list)
    rasters: list[bytes | None] = field(default_factory=list)  # parallel to records


@dataclass
class SegMap:
    results: dict[str, SegConceptResult] = field(default_factory=dict)
    blockages: list[DerivedBlockage] = field(default_factory=list)
    severity: SeverityScore = field(default_factory=lambda: SeverityScore(0.0, "minor"))
    frame_size: tuple[int, int] | None = None


Segmenter = Callable[[bytes, str], str]  # (frame_png, concept) -> output_text


def build_seg_map(
    segmenter: Segmenter, png: bytes, concepts: tuple[str, ...] | list[str], grid: int = 64
) -> SegMap:
    """Run one concept request per concept and assemble the segmentation map.

    Total: endpoint errors, empty (zero-match) outputs, and undecodable
    masks all degrade to empty records, never raise.
    """
    seg = SegMap()
    for concept in concepts:
        try:
            text = segmenter(png, concept)
        except Exception:
            text = ""
        _, records = parse_sam_output(text)
        rasters: list[bytes | None] = []
        for record in records:
            try:
                rasters.append(decode_record_mask(record))
            except Exception:
                rasters.append(None)
        seg.results[concept] = SegConceptResult(concept, records, rasters)
    for result in seg.results.values():
        if result.records:
            seg.frame_size = (result.records[0].box.w, result.records[0].box.h)
            break
    cells = {
        concept: (
            rasterize_concept(
                [(r, ras) for r, ras in zip(result.records, result.rasters) if ras is not None],
                seg.frame_size,
                grid,
            )
            if seg.frame_size
            else set()
        )
        for concept, result in seg.results.items()
    }
    seg.blockages = apply_overlap_rules(cells)
    hazard_union: set[tuple[int, int]] = set().union(
        *(cells.get(concept, set()) for concept in HAZARD_CONCEPTS)
    )
    seg.severity = severity_of(hazard_union, grid * grid)
    return seg


def _prompted_record_valid(
    label: str, png: bytes, record: SAMRecord, raster: bytes | None
) -> bool:
    """Photometric gate: prompted masks must look like their label.

    A smoke point can land on a tree crown (dotted columns have gaps);
    SAM then segments the crown. Smoke must be near-white, fire
    orange-red. Undecodable frames/pixels pass (never gate tests).
    """
    if label not in ("fire", "smoke") or raster is None:
        return True
    try:
        from PIL import Image as PILImage  # noqa: PLC0415
        import io  # noqa: PLC0415

        img = PILImage.open(io.BytesIO(png)).convert("RGB")
        px = img.load()
        w, h = img.size
    except Exception:
        return True
    box = record.box
    bw = box.x2 + 1 - box.x1
    bh = box.y2 + 1 - box.y1
    mh, mw = record.mask.height, record.mask.width
    if mh <= 0 or mw <= 0 or len(raster) != mh * mw:
        return True
    vals: list[tuple[int, int, int]] = []
    for r in range(mh):
        base = r * mw
        for c in range(mw):
            if not raster[base + c]:
                continue
            x = int(box.x1 + (c + 0.5) * bw / mw)
            y = int(box.y1 + (r + 0.5) * bh / mh)
            if 0 <= x < w and 0 <= y < h:
                vals.append(px[x, y])
                if len(vals) >= 200:
                    break
        if len(vals) >= 200:
            break
    if not vals:
        return True
    r = sum(v[0] for v in vals) / len(vals)
    g = sum(v[1] for v in vals) / len(vals)
    b = sum(v[2] for v in vals) / len(vals)
    if label == "smoke":
        return (r + g + b) / 3 > 170
    return r > 140 and r > b + 20


def build_prompted_seg_map(
    segmenter: Segmenter,
    png: bytes,
    prompts: list[tuple[str, str]],
    grid: int = 64,
) -> SegMap:
    """Seg-map from explicit prompt texts filed under `labels`.

    Same shape as build_seg_map but the caller supplies full prompt
    strings (e.g. sim-seeded point prompts) instead of bare concepts.
    Endpoint errors, empty outputs, undecodable masks, and
    photometrically implausible masks degrade to empty records, never
    raise.
    """
    seg = SegMap()
    for label, prompt in prompts:
        try:
            text = segmenter(png, prompt)
        except Exception:
            text = ""
        _, records = parse_sam_output(text)
        rasters: list[bytes | None] = []
        for record in records:
            try:
                rasters.append(decode_record_mask(record))
            except Exception:
                rasters.append(None)
        gated = [
            (rec, ras)
            for rec, ras in zip(records, rasters)
            if _prompted_record_valid(label, png, rec, ras)
        ]
        records = [rec for rec, _ in gated]
        rasters = [ras for _, ras in gated]
        prior = seg.results.get(label)
        if prior is None:
            seg.results[label] = SegConceptResult(label, records, rasters)
        else:
            prior.records.extend(records)
            prior.rasters.extend(rasters)
    for result in seg.results.values():
        if result.records:
            seg.frame_size = (result.records[0].box.w, result.records[0].box.h)
            break
    cells = {
        concept: (
            rasterize_concept(
                [(r, ras) for r, ras in zip(result.records, result.rasters) if ras is not None],
                seg.frame_size,
                grid,
            )
            if seg.frame_size
            else set()
        )
        for concept, result in seg.results.items()
    }
    seg.blockages = apply_overlap_rules(cells)
    hazard_union: set[tuple[int, int]] = set().union(
        *(cells.get(concept, set()) for concept in HAZARD_CONCEPTS)
    )
    seg.severity = severity_of(hazard_union, grid * grid)
    return seg


def seg_summary_line(seg: SegMap) -> str:
    """One-line seg-map digest for edge reports (deterministic order)."""
    counts = ",".join(f"{c}:{len(r.records)}" for c, r in seg.results.items())
    blockages = ",".join(f"{b.label}:{b.coverage:.2f}" for b in seg.blockages) or "none"
    return (
        f"segmap severity={seg.severity.label}({seg.severity.index:.2f}) "
        f"blockages=[{blockages}] counts={{{counts}}}"
    )


def max_scans_from_env(default: int = 3) -> int:
    """Per-episode SAM scan budget from DMB_SAM_MAX_SCANS_PER_EPISODE."""
    import os

    raw = os.environ.get("DMB_SAM_MAX_SCANS_PER_EPISODE", "").strip()
    if not raw:
        return default
    value = int(raw)  # invalid integers fail loudly
    if value < 1:
        raise ValueError(f"DMB_SAM_MAX_SCANS_PER_EPISODE must be >= 1 (got {value})")
    return value


def sam_from_env() -> Segmenter:
    """Build the default segmenter from DMB_SAM_BASE_URL/_API_KEY/_MODEL."""
    import os
    from functools import partial

    base_url = os.environ.get("DMB_SAM_BASE_URL", "").strip() or SAM_DEFAULT_BASE_URL
    api_key = os.environ.get("DMB_SAM_API_KEY", "").strip() or os.environ.get(
        "META_AI_API_KEY", ""
    ).strip()
    model = os.environ.get("DMB_SAM_MODEL", "").strip() or SAM_DEFAULT_MODEL
    if not api_key:
        raise RuntimeError(
            "Set DMB_SAM_API_KEY (or the shared META_AI_API_KEY) "
            "for live SAM segmentation (or inject a stub segmenter)."
        )
    return partial(openai_segmenter, model=model, base_url=base_url, api_key=api_key)


def openai_segmenter(
    png: bytes,
    concept: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float = 600.0,
) -> str:
    """Default segmenter over the Meta Model API Responses endpoint.

    Native 720p frames exceed the OpenAI default 60 s timeout; 600 s
    covers observed SAM-3.1 latency with headroom (live peat 2026-11).
    """
    import base64

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.responses.create(
        model=model,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": concept},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(png).decode(),
                    },
                ],
            }
        ],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
    )
    return response.output_text or ""


def rubble_point_prompt(
    size: tuple[int, int],
    positive: tuple[int, int],
    negatives: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> str:
    """Build an `input_text` delineating one rubble candidate.

    `size` is (width, height) in media pixels; `positive` sits on the
    rubble, `negatives` on regions to exclude (e.g. intact road).
    """
    points = [PromptPoint(*positive)]
    points.extend(PromptPoint(x, y, label="negative") for x, y in negatives)
    return build_object_prompt(
        size=(size[0], size[1]),
        objects=[PromptObject(id=1, prompts=[FramePrompt(frame=0, points=points)])],
    )


# Pixel box -> grid cell. Native backends inject a homography here once
# camera grounding lands; without one no victim detection can ground.
Grounding = Callable[[SAMRecord], tuple[int, int] | None]

# Episode state -> [(label, (x, y) pixels)]: sim-seeded point prompts.
PointProposer = Callable[[dict], list[tuple[str, tuple[int, int]]]]

# Smoke-column prompt distances out along the radial (principal point
# -> ignition pixel). Columns are dotted billboards with gaps, so two
# depths make at least one puff likely.
SMOKE_RADIAL_PX = (120, 260)


# Max burning cells prompted per scan (each yields fire + smoke points).
MAX_PROMPT_CELLS = 4

# Second fire point per cell, offset +37 u-px (~5 m): a tree can stand
# on the cell-center point (Kalimantan bed 2), and the offset still
# lands inside the 14 m bed.
FIRE_U_OFFSET_PX = 37


def ignition_point_proposer(bridge) -> PointProposer:
    """Fire/smoke point prompts for survey frames (firms scenarios).

    Open-vocab smoke/fire concepts return zero on CG renders
    (2026-11-04 probe), but sim-seeded point prompts on live burning
    cells delineate precisely (firebed 1.0%, smoke 3.9% on the
    Kalimantan nadir). Returns [] without a survey camera (native
    path untouched).
    """

    def propose(state: dict) -> list[tuple[str, tuple[int, int]]]:
        survey = getattr(bridge, "survey_camera", None)
        if survey is None:
            return []
        burning = getattr(bridge, "burning_cells", []) or []
        if not burning:
            return []
        cam, pose = survey.cam, survey.pose
        grid = state.get("grid", {})
        cell_m = getattr(bridge, "grid_cell_m", None) or 5.0
        spec = GridSpec(
            int(grid.get("width", 10)), int(grid.get("height", 10)), cell_m
        )
        out: list[tuple[str, tuple[int, int]]] = []
        for x, y in burning[:MAX_PROMPT_CELLS]:
            px = grid_to_pixel((int(x), int(y)), cam, pose, spec)
            if px is None:
                continue
            u, v = px
            out.append(("fire", (int(u), int(v))))
            ou = int(u) + FIRE_U_OFFSET_PX
            if ou < cam.width:
                out.append(("fire", (ou, int(v))))
            dx, dy = u - cam.cx, v - cam.cy
            norm = (dx * dx + dy * dy) ** 0.5 or 1.0
            for dist in SMOKE_RADIAL_PX:
                su = int(u + dx / norm * dist)
                sv = int(v + dy / norm * dist)
                if 0 <= su < cam.width and 0 <= sv < cam.height:
                    out.append(("smoke", (su, sv)))
        return out

    return propose


def hazard_point_proposer(bridge, label: str) -> PointProposer:
    """Rubble/debris point prompts for survey frames (quake/slide).

    Open-vocab damage nouns return zero on CG renders (prior finding),
    so collapse/slip-face hazard zones seed point prompts instead, one
    center + one offset point per zone. Zones gated by
    ``active_from_phase`` are skipped until their phase (aftershock
    rifts prompt only after they activate). Returns [] without a
    survey camera (native path untouched). On the sloped slide frame
    the nadir projection is approximate (flat-ground assumption);
    prompts still land inside the multi-metre debris path.
    """

    def propose(state: dict) -> list[tuple[str, tuple[int, int]]]:
        survey = getattr(bridge, "survey_camera", None)
        if survey is None:
            return []
        hazards = list(getattr(bridge, "hazards", []) or [])
        if not hazards:
            return []
        try:
            phase_idx = bridge.current_phase_index()
        except Exception:
            phase_idx = 0
        cam, pose = survey.cam, survey.pose
        grid = state.get("grid", {})
        cell_m = getattr(bridge, "grid_cell_m", None) or 5.0
        spec = GridSpec(
            int(grid.get("width", 10)), int(grid.get("height", 10)), cell_m
        )
        out: list[tuple[str, tuple[int, int]]] = []
        for hz in hazards[:MAX_PROMPT_CELLS]:
            try:
                if int(hz.get("active_from_phase", 0)) > phase_idx:
                    continue
                cx, cy = int(hz.get("x", -1)), int(hz.get("y", -1))
            except (ValueError, TypeError):
                continue
            px = grid_to_pixel((cx, cy), cam, pose, spec)
            if px is None:
                continue
            u, v = int(px[0]), int(px[1])
            out.append((label, (u, v)))
            ou = u + FIRE_U_OFFSET_PX
            if ou < cam.width:
                out.append((label, (ou, v)))
        return out

    return propose


class SAMEdgePerception(StubEdgePerception):
    """Trigger logic inherited; segmentation maps from SAM 3.1.

    Each scan captures one frame and segments the scenario's concepts.
    Victim detections need `grounding` (pixel box -> grid cell); severity
    and blockage summaries are reported regardless.
    """

    def __init__(
        self,
        bridge,  # MockBridge (duck-typed to avoid an import cycle)
        config,
        budget,
        segmenter: Segmenter,
        concepts: tuple[str, ...] | list[str] | None = None,
        grounding: Grounding | None = None,
        max_scans: int | None = None,
        seg_provider: Callable[[bytes], SegMap] | None = None,
        point_proposer: PointProposer | None = None,
        fuse_stub: bool = False,
    ) -> None:
        super().__init__(bridge, config, budget)
        self.segmenter = segmenter
        self.concepts = concepts
        self.grounding = grounding
        self.max_scans = max_scans  # None = unlimited (tests, oracle)
        # Direct SegMap source (oracle evals); bypasses segmenter+parse.
        self.seg_provider = seg_provider
        # Sim-seeded point prompts (survey frames only).
        self.point_proposer = point_proposer
        # Fuse stub radius detections under the SAM seg-map (victims stay
        # detectable where SAM has no grounding, e.g. survey frames).
        self.fuse_stub = fuse_stub
        self.last_seg_map: SegMap | None = None
        self.scans_used = 0
        self.stale = False

    def _scan(self) -> tuple[list[Detection], list]:
        self.stale = False
        drones = getattr(self.bridge, "drones", [self.bridge.drone])
        if not any(d.airborne for d in drones):
            return [], []
        if self.max_scans is not None and self.scans_used >= self.max_scans:
            # Scan budget exhausted: reuse the last seg-map, report stale
            # (fused stub detections still flow; they cost no scans).
            self.stale = True
            if self.fuse_stub:
                return StubEdgePerception._scan(self)
            return [], []
        self.last_seg_map = None
        self.scans_used += 1
        try:
            state = self.bridge.get_state()
            png = self.bridge.capture_frame(f"sam-step-{state.get('steps', 0)}")
            if png is None:
                return [], []
            # Survey cache populates on capture; read it after.
            survey = getattr(self.bridge, "survey_camera", None)
            if (
                survey is not None
                and self.point_proposer is not None
                and self.seg_provider is None
            ):
                # Survey frame: open-vocab is proven zero on CG renders,
                # so scan prompted points only (fire/smoke delineation).
                from PIL import Image as PILImage  # noqa: PLC0415
                import io  # noqa: PLC0415

                size = PILImage.open(io.BytesIO(png)).size
                prompts = [
                    (label, rubble_point_prompt(size, pt, [(8, 8)]))
                    for label, pt in self.point_proposer(state)
                ]
                seg = build_prompted_seg_map(self.segmenter, png, prompts)
            elif self.seg_provider is not None:
                seg = self.seg_provider(png)
            else:
                if self.concepts is not None:
                    concepts = list(self.concepts)
                else:
                    try:
                        concepts = list(
                            concepts_for_scenario(str(state.get("scenario", "")))
                        )
                    except ValueError:
                        concepts = list(BASE_CONCEPTS)
                seg = build_seg_map(self.segmenter, png, concepts)
        except Exception:
            return [], []
        self.last_seg_map = seg
        if self.fuse_stub:
            detections, novel = StubEdgePerception._scan(self)
        else:
            detections: list[Detection] = []
            novel: list = []
        if self.grounding is not None:
            for concept, categories in VICTIM_CONCEPTS:
                result = seg.results.get(concept)
                if result is None:
                    continue
                for record in result.records:
                    try:
                        cell = self.grounding(record)
                    except Exception:
                        continue
                    if cell is None:
                        continue
                    detection = Detection(
                        label=concept,
                        x=cell[0],
                        y=cell[1],
                        confidence=record.confidence or 0.5,
                    )
                    detections.append(detection)
                    match = self._match_victim(detection, categories)
                    if match is not None and id(match) not in self._known_ids:
                        self._known_ids.add(id(match))
                        novel.append(match)
                        self.bridge.mark_found(match)
        return detections, novel

    def _match_victim(self, detection: Detection, categories):
        """Ground a detection to the nearest same-kind victim (±1 cell)."""
        best, best_dist = None, None
        for victim in self.bridge.victims:
            if victim.category not in categories:
                continue
            dist = abs(victim.x - detection.x), abs(victim.y - detection.y)
            if dist[0] > 1 or dist[1] > 1:
                continue
            manhattan = dist[0] + dist[1]
            if best is None or manhattan < best_dist:
                best, best_dist = victim, manhattan
        return best

    def _seg_suffix(self) -> str:
        if self.last_seg_map is None:
            return ""
        suffix = f" {seg_summary_line(self.last_seg_map)}"
        if self.stale:
            suffix += " [stale seg-map: scan budget exhausted]"
        return suffix

    def on_step(self, step: int):
        report = super().on_step(step)
        if report is not None:
            report.summary = f"{report.summary}{self._seg_suffix()}"
        return report

    def on_demand(self, step: int):
        report = super().on_demand(step)
        report.summary = f"{report.summary}{self._seg_suffix()}"
        return report
