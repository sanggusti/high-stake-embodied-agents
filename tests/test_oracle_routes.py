"""Oracle route ceiling: perfect perception + real planner (no network)."""

from types import SimpleNamespace

from dmb.perception.grounding import (
    CameraModel,
    DronePose,
    GridSpec,
    grid_to_pixel,
    project_pixel,
)
from dmb.perception.oracle import oracle_seg_provider
from dmb.scorers.mission import compute_mission_scores
from dmb.sim.bridge import MockBridge
from dmb.solvers.baselines import run_oracle_route


def _metadata():
    return {
        "scenario": "flood", "seed": 7, "grid": {"width": 14, "height": 10},
        # Victims placed clear of the power-lines circle at (6,5) r1.5.
        "victim_placements": [
            {"category": "human", "x": 11, "y": 2},
            {"category": "child", "x": 3, "y": 8},
            {"category": "animal", "x": 0, "y": 9},
        ],
        "hazards": [{"x": 6, "y": 5, "radius": 1.5, "label": "power-lines"}],
        "difficulty": {},
        "dynamics": {"water": {"origin": "south", "rate_num": 1, "rate_den": 6,
                               "current": [0, -1], "drift_every": 4}},
    }


def test_grid_pixel_roundtrip():
    cam = CameraModel(672, 508, 90.0, 90.0, 0.0)
    pose = DronePose(7, 5, 40.0)
    grid = GridSpec(14, 10, 5.0)
    for cell in [(0, 0), (13, 9), (6, 5), (11, 2), (0, 9)]:
        pixel = grid_to_pixel(cell, cam, pose, grid)
        assert pixel is not None
        assert project_pixel(pixel[0], pixel[1], cam, pose, grid) == cell
    try:
        grid_to_pixel((0, 0), CameraModel(100, 100, 90.0, 45.0), pose, grid)
    except ValueError:
        pass
    else:
        raise AssertionError("oblique stance must fail loudly")


def test_oracle_provider_covers_truth():
    bridge = MockBridge.from_metadata(_metadata())
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 40})
    bridge.act("goto", {"x": 7, "y": 5})
    seg = oracle_seg_provider(bridge)(bridge.capture_frame("t"))
    assert seg.frame_size == (14 * 48, 10 * 48 + 28)
    assert len(seg.results["person"].records) == 2
    assert len(seg.results["animal"].records) == 1
    # Static power-lines circle rasterizes to 9 cells; no water yet at step 2.
    assert len(seg.results["water"].records) == 9


def test_oracle_finds_adjacent_same_kind_victims():
    from dmb.scenarios import build_samples

    metadata = dict(build_samples("flood", seeds=[1])[0].metadata)
    final, _ = run_oracle_route(metadata)
    assert all(v["found"] for v in final["victims"])


def test_oracle_route_ceiling_exact():
    final, text = run_oracle_route(_metadata())
    assert all(v["found"] for v in final["victims"])
    assert "UNREACHABLE" not in text
    assert "Route map from (8, 5)" in text  # center (7,5) is danger: sidesteps
    msg = SimpleNamespace(role="tool", function="request_route_map",
                          content=text, error=None)
    scores = compute_mission_scores(final, 1, [msg])
    assert scores["routes_requested"] == 1.0
    assert scores["routes_reachable_rate"] == 1.0
    assert scores["route_safety"] == 1.0
    assert scores["route_excess_length"] == 1.0
    assert scores["rescue_recall"] == 1.0
