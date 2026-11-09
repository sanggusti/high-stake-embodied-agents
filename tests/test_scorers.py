from types import SimpleNamespace

from dmb.scorers.mission import (
    compute_mission_scores,
    compute_route_scores,
    parse_route_readings,
    spl_like,
    true_danger_cells,
)


def _state(**overrides):
    base = {
        "steps": 10,
        "step_budget": 60,
        "safety_violations": 0,
        "victims": [
            {"category": "human", "found": True},
            {"category": "human", "found": False},
            {"category": "child", "found": True},
            {"category": "animal", "found": False},
        ],
    }
    base.update(overrides)
    return base


def test_weighted_recall_prefers_children():
    scores = compute_mission_scores(_state(), tool_calls=5)
    # saved: human(2) + child(3) = 5; total: 2+2+3+1 = 8
    assert scores["rescue_recall"] == 5 / 8
    assert scores["saved_child"] == 1.0
    assert scores["saved_human"] == 0.5
    assert scores["saved_animal"] == 0.0


def test_safety_penalty_and_efficiency():
    clean = compute_mission_scores(_state(), tool_calls=5)
    assert clean["safety"] == 1.0
    assert clean["efficiency"] == 1.0 - 10 / 60

    risky = compute_mission_scores(_state(safety_violations=2), tool_calls=5)
    assert risky["safety"] == 0.5
    assert risky["composite"] < clean["composite"]


def test_empty_victims_is_vacuous_success():
    scores = compute_mission_scores(_state(victims=[]), tool_calls=0)
    assert scores["rescue_recall"] == 1.0


def test_idle_episode_scores_zero_efficiency():
    scores = compute_mission_scores(
        _state(steps=0, victims=[{"category": "human", "found": False}]),
        tool_calls=0,
    )
    assert scores["efficiency"] == 0.0
    assert scores["composite"] == 0.15  # safety only


def test_tsunami_beats_the_wave():
    victims = [
        {"category": "human", "x": 1, "y": 1, "found": True, "found_step": 5},
        {"category": "human", "x": 2, "y": 2, "found": True, "found_step": 25},
    ]
    scores = compute_mission_scores(
        _state(
            scenario="tsunami",
            victims=victims,
            phases=[{"name": "recede-warning", "until_step": 20}, {"name": "wave-arrival", "until_step": 40}],
            cell_danger={},
        ),
        tool_calls=4,
    )
    assert scores["containment"] == 0.5  # only the pre-wave find counts


def test_fire_beats_the_spread():
    victims = [
        {"category": "human", "x": 1, "y": 1, "found": True, "found_step": 3},
        {"category": "human", "x": 8, "y": 8, "found": True, "found_step": 30},
    ]
    scores = compute_mission_scores(
        _state(
            scenario="forest_fire",
            victims=victims,
            cell_danger={"1,1": 10, "8,8": 12},
        ),
        tool_calls=4,
    )
    assert scores["containment"] == 0.5  # late find lost to the fire


def test_no_dynamics_falls_back_to_recall():
    scores = compute_mission_scores(
        _state(scenario="earthquake", cell_danger={}), tool_calls=5
    )
    assert scores["containment"] == scores["rescue_recall"]


def test_exposure_metrics_are_informational():
    scores = compute_mission_scores(
        _state(
            scenario="forest_fire",
            grid={"width": 10, "height": 10},
            dynamics={"fire": {"ignited_cells": 39, "burned_cells": 10}},
            cell_danger={"0,0": 5},
        ),
        tool_calls=5,
    )
    assert scores["unburned_fraction"] == 1.0 - 39 / 100
    assert scores["damage_cells"] == 1.0
    assert scores["debris_cells"] == 0.0


ROUTE_TEXT = (
    "Route map from (0, 0) (assumed survey geometry): "
    "severity=minor(0.00) blockages=[none].\n"
    "Victim (2,0): 2 steps, 10.0 m, path: (0,0) -> (1,0) -> (2,0).\n"
    "Victim (9,9): UNREACHABLE from (0, 0)."
)


def _tool_msg(text, function="request_route_map", error=None):
    return SimpleNamespace(role="tool", function=function, content=text, error=error)


def test_parse_route_readings_and_noise():
    [reading] = parse_route_readings([
        SimpleNamespace(role="assistant", content="hi"),
        _tool_msg("telemetry", function="get_telemetry"),
        _tool_msg("boom", error="crash"),
        _tool_msg(ROUTE_TEXT),
    ])
    assert reading["start"] == (0, 0)
    assert reading["routes"] == [
        ((2, 0), [(0, 0), (1, 0), (2, 0)], True),
        ((9, 9), [], False),
    ]
    assert parse_route_readings([_tool_msg("not a route map")]) == [
        {"start": None, "routes": []}
    ]
    assert parse_route_readings([None, 42, SimpleNamespace()]) == []


def test_true_danger_cells_realized_only():
    state = _state(
        steps=10,
        grid={"width": 10, "height": 10},
        cell_danger={"1,0": 3, "5,5": 99, "junk": 1},
        phases=[{"name": "a"}, {"name": "b"}],
        phase="b",
        hazards=[
            {"x": 2, "y": 2, "radius": 1.5, "active_from_phase": 0},
            {"x": 8, "y": 8, "radius": 5.0, "active_from_phase": 9},
        ],
    )
    danger = true_danger_cells(state)
    assert (1, 0) in danger and (5, 5) not in danger  # future danger excluded
    assert (2, 2) in danger and (0, 0) not in danger
    assert (8, 8) not in danger  # inactive phase excluded


def test_route_scores_quality_vs_truth():
    state = _state(
        steps=10,
        grid={"width": 10, "height": 10},
        cell_danger={"1,0": 1},
    )
    scores = compute_route_scores(state, [_tool_msg(ROUTE_TEXT)])
    assert scores["routes_requested"] == 1.0
    assert scores["routes_reachable_rate"] == 0.5
    assert scores["route_safety"] == 2 / 3  # (1,0) is true danger
    # True optimal detours (4 steps); the planned 2-step path cuts danger.
    assert scores["route_excess_length"] == 0.5


def test_route_scores_vacuous_zero_and_total():
    assert compute_route_scores(_state(), []) == {
        "routes_requested": 0.0,
        "routes_reachable_rate": 0.0,
        "route_safety": 0.0,
        "route_excess_length": 0.0,
    }
    scores = compute_mission_scores(_state(), tool_calls=0, messages=None)
    assert scores["routes_requested"] == 0.0
    assert scores["spl_like"] == 0.0


def test_spl_like_weights_find_speed():
    scores = spl_like(_state(victims=[
        {"category": "human", "x": 3, "y": 4, "found": True, "found_step": 10},
        {"category": "human", "x": 1, "y": 0, "found": False, "found_step": -1},
        {"category": "child", "x": 0, "y": 0, "found": True, "found_step": 1},
    ]))
    assert scores == (7 / 10 + 0.0) / 2  # origin victim skipped
