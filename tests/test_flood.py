"""Flood harder envs: concepts, survey frames, proposers."""

import pytest

from hsea.perception.render_frames import frames_roots, survey_frame
from hsea.perception.sam import concepts_for_scenario, hazard_point_proposer
from hsea.scenarios import build_samples
from hsea.sim.bridge import MockBridge


def test_hard_concepts_include_water():
    for scen in ("flood_hard", "flash_flood_hard", "tsunami_hard"):
        concepts = concepts_for_scenario(scen)
        assert "water" in concepts  # base flood/flash/tsunami extras
        assert "building" in concepts


def test_frames_roots_cover_flood():
    roots = frames_roots()
    assert any(r.name == "renders" and r.parent.name == "flood-sim2real" for r in roots)
    if survey_frame("flood_hard", 7) is None:
        pytest.skip("survey corpus not mounted (HSEA_RENDER_FRAMES_DIR)")
    flood = survey_frame("flood_hard", 7)
    flash = survey_frame("flash_flood_hard", 7)
    tsunami = survey_frame("tsunami_hard", 7)
    assert flood is not None and flash is not None and tsunami is not None
    assert (flood.cam.width, flood.cam.height) == (1280, 720)
    assert flood.pose.altitude_m == 300.0
    assert flash.pose.altitude_m == 300.0
    assert tsunami.pose.altitude_m == 320.0


def _bridged(scenario):
    if survey_frame(scenario, 7) is None:
        pytest.skip("survey corpus not mounted (HSEA_RENDER_FRAMES_DIR)")
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
