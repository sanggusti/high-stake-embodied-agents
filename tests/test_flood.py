"""Flood harder envs: concepts, survey frames, proposers, builders."""

import pathlib
import xml.etree.ElementTree as ET

from dmb.perception.render_frames import frames_roots, survey_frame
from dmb.perception.sam import concepts_for_scenario, hazard_point_proposer
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge


def test_hard_concepts_include_water():
    for scen in ("flood_hard", "flash_flood_hard", "tsunami_hard"):
        concepts = concepts_for_scenario(scen)
        assert "water" in concepts  # base flood/flash/tsunami extras
        assert "building" in concepts


def test_frames_roots_cover_flood():
    roots = frames_roots()
    assert any(r.name == "renders" and r.parent.name == "flood-sim2real" for r in roots)
    flood = survey_frame("flood_hard", 7)
    flash = survey_frame("flash_flood_hard", 7)
    tsunami = survey_frame("tsunami_hard", 7)
    assert flood is not None and flash is not None and tsunami is not None
    assert (flood.cam.width, flood.cam.height) == (1280, 720)
    assert flood.pose.altitude_m == 300.0
    assert flash.pose.altitude_m == 300.0
    assert tsunami.pose.altitude_m == 320.0


def _bridged(scenario):
    md = dict(build_samples(scenario, seeds=[7])[0].metadata)
    bridge = MockBridge.from_metadata(md)
    assert bridge.capture_frame("test") is not None
    assert bridge.survey_camera is not None
    return bridge


def test_water_proposer_seeds_in_frame_points():
    for scen in ("flood_hard", "flash_flood_hard", "tsunami_hard"):
        bridge = _bridged(scen)
        pts = hazard_point_proposer(bridge, "water")(bridge.get_state())
        assert len(pts) == 8, f"{scen}: 4 zones x (center + offset)"
        assert all(label == "water" for label, _ in pts)
        cam = bridge.survey_camera.cam
        for _, (u, v) in pts:
            assert 0 <= u < cam.width and 0 <= v < cam.height
        bridge.close()


def test_build_flood_world(tmp_path):
    from assets.structures.build_flood import build

    stats = build("flood_hard", 7, tmp_path)
    ET.fromstring((tmp_path / "world.sdf").read_text())
    assert stats["includes"] > 100
    assert stats["victim_actors"] == 4
    tex = tmp_path / "models" / "ground_tex" / "materials" / "textures"
    assert (tex / "asphalt_02.jpg").exists()
    school_sdf = (tmp_path / "models" / "school" / "model.sdf").read_text()
    assert "albedo_map" in school_sdf
    randy_sdf = (tmp_path / "models" / "rescue-randy-sitting" / "model.sdf").read_text()
    assert "fuel.ignitionrobotics.org" not in randy_sdf
    assert not (tmp_path / "models" / "wam-v").exists()  # pinned as-is
    assert (tmp_path / "models" / "flood-rain" / "model.sdf").exists()


def test_build_flash_world(tmp_path):
    from assets.structures.build_flood import build

    stats = build("flash_flood_hard", 7, tmp_path)
    ET.fromstring((tmp_path / "world.sdf").read_text())
    assert stats["includes"] > 80
    assert stats["victim_actors"] == 5
    tex = tmp_path / "models" / "ground_tex" / "materials" / "textures"
    assert (tex / "sand_01.jpg").exists()
    bridge_sdf = (tmp_path / "models" / "truss-bridge" / "model.sdf").read_text()
    assert "albedo_map" in bridge_sdf


def test_build_tsunami_world(tmp_path):
    from assets.structures.build_flood import build

    stats = build("tsunami_hard", 7, tmp_path)
    ET.fromstring((tmp_path / "world.sdf").read_text())
    assert stats["includes"] > 50
    assert stats["victim_actors"] == 4
    world = (tmp_path / "world.sdf").read_text()
    assert "model://coast-water" not in world  # retired: giant + blue
    assert "model://pier" in world and "model://dock" in world
    pier_sdf = (tmp_path / "models" / "pier" / "model.sdf").read_text()
    assert "albedo_map" in pier_sdf
