"""Quake/slide harder envs: concepts, survey frames, proposers, builders."""

import pathlib
import xml.etree.ElementTree as ET

from dmb.perception.render_frames import frames_roots, survey_frame
from dmb.perception.sam import (
    HAZARD_CONCEPTS,
    build_prompted_seg_map,
    concepts_for_scenario,
    hazard_point_proposer,
)
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge

FIX = pathlib.Path(__file__).parent / "fixtures"


def test_hard_concepts_resolve_to_base():
    quake = concepts_for_scenario("earthquake_hard")
    assert "tent" in quake  # earthquake extras
    assert "building" in quake
    slide = concepts_for_scenario("landslide_hard")
    assert slide == ("road", "building", "person", "vehicle", "car", "animal")


def test_hazard_labels_feed_severity():
    assert "rubble" in HAZARD_CONCEPTS
    assert "debris" in HAZARD_CONCEPTS
    text = (FIX / "sam_point_rubble.txt").read_text()
    seg = build_prompted_seg_map(lambda _png, _prompt: text, b"fake", [("rubble", "p")])
    assert len(seg.results["rubble"].records) == 1
    assert seg.severity.index > 0.0


def test_frames_roots_cover_quake_slide():
    roots = frames_roots()
    assert any(r.name == "renders" and r.parent.name == "quake-slide-sim2real" for r in roots)
    quake = survey_frame("earthquake_hard", 7)
    slide = survey_frame("landslide_hard", 7)
    assert quake is not None and slide is not None
    assert (quake.cam.width, quake.cam.height) == (1280, 720)
    assert quake.pose.altitude_m == 300.0
    assert slide.pose.altitude_m == 247.0


def _bridged(scenario):
    md = dict(build_samples(scenario, seeds=[7])[0].metadata)
    bridge = MockBridge.from_metadata(md)
    assert bridge.capture_frame("test") is not None
    assert bridge.survey_camera is not None
    return bridge


def test_hazard_proposer_seeds_in_frame_points():
    bridge = _bridged("earthquake_hard")
    pts = hazard_point_proposer(bridge, "rubble")(bridge.get_state())
    assert pts, "core zones must propose"
    assert all(label == "rubble" for label, _ in pts)
    cam = bridge.survey_camera.cam
    for _, (u, v) in pts:
        assert 0 <= u < cam.width and 0 <= v < cam.height
    bridge.close()
    bridge = _bridged("landslide_hard")
    pts = hazard_point_proposer(bridge, "debris")(bridge.get_state())
    assert len(pts) == 8  # 4 zones x (center + offset)
    bridge.close()


def test_hazard_proposer_respects_phases():
    md = dict(build_samples("earthquake_hard", seeds=[7])[0].metadata)
    md["hazards"] = [
        {"x": 30, "y": 30, "radius": 6.0, "label": "core"},
        {"x": 20, "y": 15, "radius": 5.0, "label": "rift", "active_from_phase": 1},
    ]
    bridge = MockBridge.from_metadata(md)
    bridge.capture_frame("test")
    early = hazard_point_proposer(bridge, "rubble")(bridge.get_state())
    assert [pt for _, pt in early] and len(early) == 2  # core only
    bridge.steps = 41  # past main-shock until_step 40
    late = hazard_point_proposer(bridge, "rubble")(bridge.get_state())
    assert len(late) == 4  # core + rift
    bridge.close()


def test_hazard_proposer_without_survey_is_empty():
    md = dict(build_samples("flood", seeds=[1])[0].metadata)
    bridge = MockBridge.from_metadata(md)
    assert hazard_point_proposer(bridge, "rubble")(bridge.get_state()) == []
    bridge.close()


def test_build_quake_world(tmp_path):
    from assets.structures.build_quake import build

    stats = build("earthquake_hard", 7, tmp_path)
    ET.fromstring((tmp_path / "world.sdf").read_text())
    assert stats["includes"] > 50
    assert stats["victim_actors"] == 4
    assert (tmp_path / "models" / "ground_tex" / "materials" / "textures" / "asphalt_02.jpg").exists()
    house_sdf = (tmp_path / "models" / "house-1" / "model.sdf").read_text()
    assert "albedo_map" in house_sdf and "<script>" not in house_sdf


def test_build_slide_world(tmp_path):
    from assets.structures.build_slide import build

    stats = build("landslide_hard", 7, tmp_path)
    ET.fromstring((tmp_path / "world.sdf").read_text())
    assert stats["includes"] > 150
    assert stats["victim_actors"] == 4
    assert (tmp_path / "models" / "ground_tex" / "materials" / "textures" / "brown_mud_02.jpg").exists()
    assert (tmp_path / "models" / "slide-rain" / "model.sdf").exists()
