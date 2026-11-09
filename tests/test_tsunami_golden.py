"""Tsunami golden: scripted rollout + phase timing regression tests."""

from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.scenarios import build_samples
from dmb.scorers.mission import compute_mission_scores
from dmb.sim.bridge import MockBridge


def _rollout(seed: int = 1) -> tuple[MockBridge, dict]:
    meta = build_samples("tsunami", seeds=[seed])[0].metadata
    bridge = MockBridge.from_metadata(dict(meta))
    edge = StubEdgePerception(
        bridge, TriggerConfig(heartbeat_every_steps=1000), FrameBudget(8)
    )

    def do(action: str, params: dict | None = None):
        obs = bridge.act(action, params or {})
        edge.on_step(bridge.get_state()["steps"])
        return obs

    do("arm")
    do("takeoff", {"altitude_m": 10})
    for victim in meta["victim_placements"][:2]:
        do("goto", {"x": victim["x"], "y": victim["y"]})
    do("loiter", {"seconds": 5})
    do("land")
    return bridge, compute_mission_scores(bridge.get_state(), 6)


def test_tsunami_seed_1_golden():
    bridge, scores = _rollout()
    assert bridge.get_state()["phase"] == "recede-warning"
    assert scores == {
        "saved_human": 2 / 3,
        "saved_child": 0.5,
        "saved_animal": 0.0,
        "rescue_recall": 7 / 13,
        "containment": 7 / 13,
        "efficiency": 11 / 12,
        "safety": 1.0,
        "tool_calls": 6.0,
        "found_any": 1.0,
        "first_find_step": 2.0,
        "coverage": 3 / 120,
        "damage_cells": 120.0,
        "unburned_fraction": 1.0,
        "unflooded_fraction": 0.9,
        "debris_cells": 0.0,
        "buried_victims": 0.0,
        "routes_requested": 0.0,
        "routes_reachable_rate": 0.0,
        "route_safety": 0.0,
        "route_excess_length": 0.0,
        "spl_like": 3 / 6,  # 3 found at ≤manhattan steps, 3 missed
        "composite": 0.5 * (7 / 13)
        + 0.2 * (7 / 13)
        + 0.15 * (11 / 12)
        + 0.15 * 1.0,
    }


def test_phases_advance_with_steps():
    bridge = MockBridge.from_metadata(build_samples("tsunami", seeds=[1])[0].metadata)
    assert bridge.current_phase() == "recede-warning"
    bridge.steps = 20
    assert bridge.current_phase() == "wave-arrival"
    bridge.steps = 40
    assert bridge.current_phase() == "debris"
    bridge.steps = 999
    assert bridge.current_phase() == "debris"


def _phased_metadata():
    return {
        "scenario": "test",
        "seed": 0,
        "grid": {"width": 10, "height": 10},
        "victim_placements": [],
        "hazards": [
            {"x": 6, "y": 2, "radius": 3.0, "label": "surge", "active_from_phase": 1},
            {"x": 9, "y": 6, "radius": 1.5, "label": "debris", "active_from_phase": 2},
        ],
        "difficulty": {
            "phases": [
                {"name": "p0", "until_step": 20},
                {"name": "p1", "until_step": 40},
                {"name": "p2", "until_step": 60},
            ]
        },
    }


def test_hazards_activate_by_phase():
    bridge = MockBridge.from_metadata(_phased_metadata())
    # Surge sleeps during p0, live at p1.
    assert bridge._in_hazard(6, 2) is False
    bridge.steps = 25
    assert bridge._in_hazard(6, 2) is True
    # Debris sleeps until p2.
    assert bridge._in_hazard(9, 6) is False
    bridge.steps = 45
    assert bridge._in_hazard(9, 6) is True


def test_tsunami_water_adds_dynamic_danger():
    meta = build_samples("tsunami", seeds=[1])[0].metadata
    bridge = MockBridge.from_metadata(dict(meta))
    # (9,6) floods at step 16, before its static debris phase (40).
    assert bridge._in_hazard(9, 6) is False
    bridge.steps = 20
    assert bridge._in_hazard(9, 6) is True


def test_scenarios_without_phases_stay_nominal():
    bridge = MockBridge.from_metadata(build_samples("forest_fire", seeds=[1])[0].metadata)
    assert bridge.current_phase() == "nominal"
    assert bridge._in_hazard(7, 7) is True
