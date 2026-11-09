"""SAM 3.1 seg-maps: parser, concepts, overlaps, severity, edge flow (stubbed).

Fixtures are real `sam-3.1` outputs captured 2026-11-02 (see docs/sam-segmap.md);
no test in this file touches the network.
"""

import os
import pathlib

from dmb.perception.sam import (
    SAMBox,
    SAMEdgePerception,
    SAMMask,
    SAMRecord,
    apply_overlap_rules,
    build_seg_map,
    concepts_for_scenario,
    decode_record_mask,
    mask_fill,
    max_scans_from_env,
    parse_sam_output,
    parse_sam_video_output,
    rasterize_concept,
    record_frame_fraction,
    rubble_point_prompt,
    sam_from_env,
    severity_of,
)
from dmb.perception.edge import Detection, FrameBudget, TriggerConfig
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge

FIX = pathlib.Path(__file__).parent / "fixtures"


def _rec(x1, y1, x2, y2, w=10, h=10, c=0.9, H=2, W=2):  # noqa: N803
    return SAMRecord(
        object_id=0,
        box=SAMBox(x1, y1, x2, y2, w, h, c),
        mask=SAMMask(H, W, "lossless", "~fake"),
    )


def test_parse_point_prompt_fixture():
    frame, records = parse_sam_output((FIX / "sam_point_rubble.txt").read_text())
    assert frame == 0
    assert len(records) == 1
    rec = records[0]
    assert rec.object_id == 1
    assert (rec.box.x1, rec.box.y1, rec.box.x2, rec.box.y2) == (517, 337, 684, 392)
    assert (rec.box.w, rec.box.h) == (1280, 852)
    assert rec.confidence == 0.90625
    assert (rec.mask.height, rec.mask.width, rec.mask.encoding) == (56, 168, "one_bit")
    raster = decode_record_mask(rec)
    assert len(raster) == 56 * 168
    assert abs(mask_fill(raster) - 0.627) < 0.01


def test_parse_multi_record_person_fixture():
    frame, records = parse_sam_output((FIX / "sam_person.txt").read_text())
    assert frame == 0
    assert [r.object_id for r in records] == [0, 1]
    assert records[0].confidence == 0.8552036285400391
    raster = decode_record_mask(records[0])
    frac = record_frame_fraction(records[0], raster)
    assert 0.0 < frac < 0.01  # small figure in a large frame


def test_parse_empty_and_malformed():
    assert parse_sam_output("") == (None, [])
    assert parse_sam_output("not a sam line") == (None, [])
    frame, records = parse_sam_output(
        "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|>"
        "<|mask;x=0;y=0;data=2,2,~AAAAAA|>,1<|box;oops|>"
    )
    assert frame == 0
    assert len(records) == 1  # malformed record dropped, good one kept


def test_parse_video_frames_keyed_by_marker():
    line = (
        "<{f}f>0<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|>"
        "<|mask;x=0;y=0;data=2,2,~AAAAAA|>"
    )
    got = parse_sam_video_output(line.format(f=0) + "\n" + line.format(f=2) + "\n")
    assert sorted(got) == [0, 2]  # skipped frame 1 stays skipped
    assert len(got[2]) == 1


def test_concepts_for_scenario():
    assert concepts_for_scenario("flood") == (
        "road", "building", "person", "vehicle", "car", "animal", "water")
    assert "smoke" in concepts_for_scenario("forest_fire_large")
    assert "tent" in concepts_for_scenario("earthquake")
    try:
        concepts_for_scenario("atlantis")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown scenario must fail loudly")


def test_overlap_flooded_road():
    # 10x10 frame at grid 10: road fills rows 5..9, water fills cols 0..4.
    road = _rec(0, 5, 9, 9, H=5, W=10)
    water = _rec(0, 0, 4, 9, H=10, W=5)
    cells = {
        "road": rasterize_concept([(road, bytes([1]) * 50)], (10, 10), grid=10),
        "water": rasterize_concept([(water, bytes([1]) * 50)], (10, 10), grid=10),
    }
    assert len(cells["road"]) == 50
    [block] = apply_overlap_rules(cells)
    assert block.label == "flooded_road"
    assert block.coverage == 0.5
    assert block.cells == 25


def test_overlap_needs_both_sides_and_min_coverage():
    road = _rec(0, 0, 9, 9, H=10, W=10)
    cells = {"road": rasterize_concept([(road, bytes([1]) * 100)], (10, 10), grid=10)}
    assert apply_overlap_rules(cells) == []  # no water, no blockage
    drop = _rec(0, 0, 0, 0, H=1, W=1)
    cells["water"] = rasterize_concept([(drop, bytes([1]))], (10, 10), grid=10)
    assert apply_overlap_rules(cells) == []  # 1% overlap is noise


def test_severity_labels():
    assert severity_of(set(), 100).label == "minor"
    assert severity_of(set(range(4)), 100).label == "minor"
    assert severity_of(set(range(5)), 100).label == "moderate"
    assert severity_of(set(range(19)), 100).label == "moderate"
    assert severity_of(set(range(20)), 100).label == "severe"


def test_rubble_point_prompt_exact():
    assert rubble_point_prompt((1280, 852), (640, 380), [(760, 505)]) == (
        "<0f>1<|point;x=640;y=380;w=1280;h=852|>"
        "-<|point;x=760;y=505;w=1280;h=852|>"
    )


def test_sam_from_env_key_and_fallback():
    old = dict(os.environ)
    try:
        os.environ.pop("DMB_SAM_API_KEY", None)
        os.environ.pop("META_AI_API_KEY", None)
        try:
            sam_from_env()
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing key must fail loudly")
        os.environ["META_AI_API_KEY"] = "fallback-key"
        seg = sam_from_env()
        assert seg.keywords["api_key"] == "fallback-key"
        assert seg.keywords["base_url"] == "https://api.meta.ai/v1"
        assert seg.keywords["model"] == "sam-3.1"
        os.environ["DMB_SAM_API_KEY"] = " direct-key "
        assert sam_from_env().keywords["api_key"] == "direct-key"
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_build_seg_map_zero_matches():
    seg = build_seg_map(lambda png, concept: "", b"png", ("road", "water"))
    assert seg.results["road"].records == []
    assert seg.blockages == []
    assert (seg.severity.label, seg.severity.index) == ("minor", 0.0)
    assert seg.frame_size is None


def _person_line():
    return (
        "<0f>0<|box;x1=10;y1=10;x2=20;y2=20;w=100;h=100;c=0.9|>"
        "<|mask;x=0;y=0;c=0.9;data=2,2,~AAAAAA|>"
    )


def test_sam_edge_novelty_flow_with_stub():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    victim = bridge.victims[0]

    def segmenter(png: bytes, concept: str) -> str:
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        return _person_line() if concept == "person" else ""

    edge = SAMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8), segmenter,
        grounding=lambda rec: (victim.x, victim.y),
    )
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    bridge.act("goto", {"x": victim.x, "y": victim.y})
    report = edge.on_step(bridge.get_state()["steps"])
    assert report is not None and report.trigger == "novelty"
    assert bridge.get_state()["victims"][0]["found"] is True
    assert "segmap" in report.summary


def test_sam_edge_without_grounding_reports_no_victims():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    edge = SAMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8),
        lambda png, concept: _person_line(),
    )
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    report = edge.on_demand(bridge.get_state()["steps"])
    assert report.detections == []  # pixels cannot ground to grid without a camera model
    assert "segmap" in report.summary
    assert edge.last_seg_map is not None


def test_max_scans_from_env():
    old = dict(os.environ)
    try:
        os.environ.pop("DMB_SAM_MAX_SCANS_PER_EPISODE", None)
        assert max_scans_from_env() == 3
        os.environ["DMB_SAM_MAX_SCANS_PER_EPISODE"] = "7"
        assert max_scans_from_env() == 7
        for bad in ("0", "-2", "lots"):
            os.environ["DMB_SAM_MAX_SCANS_PER_EPISODE"] = bad
            try:
                max_scans_from_env()
            except ValueError:
                pass
            else:
                raise AssertionError(f"must reject {bad!r}")
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_scan_budget_reuses_last_map_and_marks_stale():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    calls = []

    def segmenter(png: bytes, concept: str) -> str:
        calls.append(concept)
        return _person_line() if concept == "person" else ""

    edge = SAMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8), segmenter, max_scans=1)
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    first = edge.on_demand(bridge.get_state()["steps"])
    assert "stale" not in first.summary
    assert edge.scans_used == 1
    kept = edge.last_seg_map
    second = edge.on_demand(bridge.get_state()["steps"])
    assert edge.scans_used == 1  # no new scan
    assert edge.last_seg_map is kept  # reuse, not empty
    assert "[stale seg-map: scan budget exhausted]" in second.summary
    assert len(calls) == len(concepts_for_scenario("forest_fire"))


def test_match_victim_prefers_nearest():
    bridge = MockBridge.from_metadata({
        "scenario": "flood", "seed": 0, "grid": {"width": 14, "height": 10},
        "victim_placements": [
            {"category": "human", "x": 7, "y": 7},
            {"category": "human", "x": 7, "y": 6},
        ],
        "hazards": [], "difficulty": {}, "dynamics": {},
    })
    edge = SAMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8), lambda png, c: "")
    assert edge._match_victim(
        Detection("person", 7, 6, 0.9), ("human", "child")).y == 6
    assert edge._match_victim(
        Detection("person", 7, 7, 0.9), ("human", "child")).y == 7
    assert edge._match_victim(
        Detection("person", 0, 0, 0.9), ("human", "child")) is None


def test_sam_edge_failure_degrades_quietly():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))

    def boom(png: bytes, concept: str) -> str:
        raise RuntimeError("endpoint down")

    edge = SAMEdgePerception(bridge, TriggerConfig(), FrameBudget(8), boom)
    assert edge.on_step(0) is None
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    assert edge.on_step(bridge.get_state()["steps"]) is None
    report = edge.on_demand(bridge.get_state()["steps"])
    assert report.detections == []


def test_concepts_firms_includes_smoke():
    concepts = concepts_for_scenario("firms_peat_kalimantan_01")
    assert "smoke" in concepts and "person" in concepts


def test_build_prompted_seg_map_labels():
    from dmb.perception.sam import build_prompted_seg_map

    seg = build_prompted_seg_map(
        lambda png, prompt: _person_line(), b"png", [("fire", "prompt1")]
    )
    assert len(seg.results["fire"].records) == 1
    assert seg.frame_size == (100, 100)
    assert seg.severity.index >= 0.0


def test_ignition_point_proposer_hits_firebed():
    from dmb.perception.grounding import CameraModel, DronePose
    from dmb.perception.render_frames import SurveyFrame
    from dmb.perception.sam import ignition_point_proposer

    class FakeBridge:
        survey_camera = SurveyFrame(
            b"png",
            CameraModel(1280, 720, 60.0, 90.0, 90.0),
            DronePose(59, 63, 150.0),
        )
        grid_cell_m = 5.0
        burning_cells = [(58, 60)]

    propose = ignition_point_proposer(FakeBridge())
    state = {"grid": {"width": 120, "height": 120}}
    points = propose(state)
    fire = [pt for label, pt in points if label == "fire"]
    smoke = [pt for label, pt in points if label == "smoke"]
    assert len(fire) == 2 and len(smoke) == 2  # offset fire + two depths
    assert fire[1][0] - fire[0][0] == 37
    assert abs(fire[0][0] - 732) < 40
    assert abs(fire[0][1] - 379) < 40
    cx, cy = 640, 360
    d_fire = abs(fire[0][0] - cx) + abs(fire[0][1] - cy)
    for pt in smoke:
        assert abs(pt[0] - cx) + abs(pt[1] - cy) > d_fire


def test_fuse_stub_merges_detections():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    victim = bridge.victims[0]
    edge = SAMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8),
        lambda png, concept: "",
        fuse_stub=True,
    )
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    bridge.act("goto", {"x": victim.x, "y": victim.y})
    report = edge.on_step(bridge.get_state()["steps"])
    assert report is not None and report.detections != []
    assert edge.last_seg_map is not None  # SAM side still scanned


def test_stale_fuse_keeps_stub_detections():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    victim = bridge.victims[0]
    edge = SAMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8),
        lambda png, concept: "",
        max_scans=0,
        fuse_stub=True,
    )
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    bridge.act("goto", {"x": victim.x, "y": victim.y})
    detections, _ = edge._scan()
    assert edge.stale is True
    assert detections != []


def test_survey_frame_loads_kalimantan():
    from dmb.perception.render_frames import survey_frame

    survey = survey_frame("firms_peat_kalimantan_01", 7)
    assert survey is not None
    assert survey.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (survey.cam.width, survey.cam.height) == (1280, 720)
    assert (survey.cam.pitch_deg, survey.cam.yaw_deg) == (90.0, 90.0)
    assert survey_frame("forest_fire", 1) is None


def test_bridge_capture_serves_survey_frame():
    from dmb.perception.render_frames import survey_frame

    meta = dict(build_samples("firms_peat_kalimantan_01", seeds=[7])[0].metadata)
    bridge = MockBridge.from_metadata(meta)
    png = bridge.capture_frame("test")
    assert png == survey_frame("firms_peat_kalimantan_01", 7).png
    assert bridge.survey_camera is not None


def test_prompted_photometric_gate():
    import io

    from PIL import Image

    from dmb.perception.sam import _prompted_record_valid

    buf = io.BytesIO()
    frame = Image.new("RGB", (100, 100), (250, 250, 250))
    frame.paste(Image.new("RGB", (20, 20), (40, 90, 40)), (10, 10))  # crown
    frame.save(buf, format="PNG")
    png = buf.getvalue()
    crown = _rec(10, 10, 29, 29, H=2, W=2)
    white = _rec(60, 60, 79, 79, H=2, W=2)
    full = bytes([1, 1, 1, 1])
    assert _prompted_record_valid("smoke", png, white, full) is True
    assert _prompted_record_valid("smoke", png, crown, full) is False
    assert _prompted_record_valid("fire", png, crown, full) is False
    assert _prompted_record_valid("smoke", b"not-a-png", crown, full) is True
