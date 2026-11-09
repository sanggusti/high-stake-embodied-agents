"""Phase B: dynamics snapshots, plausibility ordering, containment monotonicity."""

from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.physics.fire import FireModel
from dmb.physics.terrain import SlipModel
from dmb.physics.water import WaterModel
from dmb.scorers.mission import compute_mission_scores
from dmb.sim.bridge import MockBridge, Victim

FIRE_CFG = {
    "ignition": [{"x": 2, "y": 2}],
    "moisture": 0.3,
    "wind_dir": [1, 0],
    "wind_speed": 3.0,
    "fuel_default": 0.7,
    "burn_duration": 25,
    "ignition_threshold": 1.0,
    "base_ros": 0.06,
}


def _run_fire(config, width=6, height=6, steps=30):
    model = FireModel(config, width, height)
    for step in range(1, steps + 1):
        model.step(step)
    return model


def test_fire_snapshot_downwind_bias():
    model = _run_fire(FIRE_CFG)
    assert model.snapshot()["ignited"] == ["2,2@0", "3,1@27", "3,2@24", "3,3@27"]


def test_wind_increases_spread():
    calm = _run_fire({**FIRE_CFG, "wind_speed": 0.0})
    windy = _run_fire({**FIRE_CFG, "wind_speed": 6.0})
    assert len(windy.ignited) > len(calm.ignited)


def test_moisture_decreases_spread():
    dry = _run_fire({**FIRE_CFG, "moisture": 0.1})
    wet = _run_fire({**FIRE_CFG, "moisture": 0.8})
    assert len(dry.ignited) > len(wet.ignited)


def test_scenario_fire_stays_bounded():
    model = FireModel(
        {
            "ignition": [{"x": 7, "y": 7}, {"x": 3, "y": 8}],
            "moisture": 0.3,
            "wind_dir": [1, 0],
            "wind_speed": 3.0,
            "fuel_default": 0.7,
            "burn_duration": 25,
            "ignition_threshold": 1.0,
            "base_ros": 0.06,
        },
        10,
        10,
    )
    for step in range(1, 61):
        model.step(step)
    assert 20 < len(model.ignited) < 60, len(model.ignited)


def test_water_flood_snapshot_and_schedule():
    model = WaterModel(
        {"origin": "south", "rate_num": 1, "rate_den": 4, "current": [0, -1], "drift_every": 3},
        12,
        10,
    )
    for step in range(1, 13):
        new = model.step(step)
    assert model.flooded_at(12) == {(x, y) for x in range(12) for y in range(7, 10)}
    assert new == {(x, 7) for x in range(12)}
    assert model.flood_step((5, 9)) == 4
    assert model.flood_step((5, 0)) == 40


def test_water_drifts_only_mobile_flooded_victims():
    model = WaterModel(
        {"origin": "south", "rate_num": 1, "rate_den": 1, "current": [0, -1], "drift_every": 3},
        10,
        10,
    )
    for step in range(1, 10):
        model.step(step)
    floater = Victim(category="animal", x=5, y=8, mobility="mobile")
    static = Victim(category="human", x=5, y=8, mobility="static")
    assert model.drift(floater, 9) is True
    assert (floater.x, floater.y) == (5, 7)
    assert model.drift(static, 9) is False
    dry = Victim(category="animal", x=5, y=0, mobility="mobile")
    assert model.drift(dry, 9) is False


def test_slip_snapshot_and_burial():
    model = SlipModel(
        {
            "candidates": [{"x": 5, "y": 8, "slope": 1.5}, {"x": 4, "y": 7, "slope": 1.0}],
            "sat_rate": 0.05,
            "rain_multiplier": 2.0,
            "raining": True,
            "threshold": 1.0,
            "debris_dir": [0, 1],
            "debris_len": 3,
        },
        10,
        12,
    )
    for step in range(1, 61):
        model.step(step)
    assert model.snapshot() == {
        "slip_step": 11,
        "debris": ["5,10", "5,8", "5,9"],
        "saturation": 1.1,
    }


def test_slip_stability_rules():
    dry_cfg = {
        "candidates": [{"x": 1, "y": 1, "slope": 2.0}],
        "sat_rate": 0.0,
        "threshold": 1.0,
        "debris_dir": [0, 1],
        "debris_len": 2,
    }
    dry = SlipModel(dry_cfg, 10, 10)
    for step in range(1, 61):
        assert dry.step(step) == set()
    assert dry.slip_step is None

    steep = SlipModel(
        {**dry_cfg, "sat_rate": 1.0, "raining": False, "threshold": 0.5}, 10, 10
    )
    assert steep.step(1) == {(1, 1), (1, 2)}
    assert steep.slip_step == 1
    # one slip per episode
    assert steep.step(2) == set()


def _beat_spread_rollout(delay: int) -> dict:
    fire_cfg = {
        "ignition": [{"x": 7, "y": 7}],
        "moisture": 0.3,
        "wind_dir": [1, 0],
        "wind_speed": 3.0,
        "fuel_default": 0.7,
        "burn_duration": 25,
        "ignition_threshold": 1.0,
        "base_ros": 0.06,
    }
    metadata = {
        "scenario": "test",
        "seed": 0,
        "grid": {"width": 10, "height": 10},
        "victim_placements": [{"category": "human", "x": 8, "y": 7}],
        "hazards": [],
        "difficulty": {},
        "dynamics": {"fire": fire_cfg},
    }
    bridge = MockBridge.from_metadata(metadata)
    edge = StubEdgePerception(bridge, TriggerConfig(heartbeat_every_steps=1000), FrameBudget(8))

    def do(action: str, params: dict | None = None):
        obs = bridge.act(action, params or {})
        edge.on_step(bridge.get_state()["steps"])

    do("arm")
    do("takeoff", {"altitude_m": 10})
    for _ in range(delay):
        do("loiter", {"seconds": 1})
    do("goto", {"x": 8, "y": 7})
    do("land")
    return compute_mission_scores(bridge.get_state(), 0)


def test_containment_monotonic_with_delay():
    fast = _beat_spread_rollout(delay=0)
    slow = _beat_spread_rollout(delay=30)
    assert fast["rescue_recall"] == slow["rescue_recall"] == 1.0
    assert fast["containment"] == 1.0
    assert slow["containment"] == 0.0


def test_fire_weather_diverges_same_seed():
    import pytest

    mods = {
        "weather_modifiers": {
            "calm-wet": {"wind_speed": 1.0, "moisture": 0.45},
            "windy-dry": {"wind_speed": 6.0, "moisture": 0.15},
        }
    }
    calm = _run_fire({**FIRE_CFG, **mods, "weather_active": "calm-wet"})
    windy = _run_fire({**FIRE_CFG, **mods, "weather_active": "windy-dry"})
    assert calm.snapshot()["ignited"] != windy.snapshot()["ignited"]
    assert len(windy.ignited) > len(calm.ignited)
    repeat = _run_fire({**FIRE_CFG, **mods, "weather_active": "windy-dry"})
    assert repeat.snapshot() == windy.snapshot()
    with pytest.raises(ValueError, match="Unknown fire weather"):
        FireModel({**FIRE_CFG, **mods, "weather_active": "storm"}, 6, 6)


def test_weather_axis_end_to_end():
    from dmb.scenarios import build_samples

    plain = build_samples("forest_fire", seeds=[1])[0]
    assert plain.metadata["weather_active"] is None
    wet = build_samples("forest_fire", seeds=[1], weather="haze-heavy")[0]
    assert wet.id == "forest_fire-seed-1-haze-heavy"
    dry = build_samples("forest_fire", seeds=[1], weather="clear")[0]
    b_wet, b_dry = MockBridge(), MockBridge()
    b_wet.reset(dict(wet.metadata))
    b_dry.reset(dict(dry.metadata))
    assert b_wet._fire.wind_speed == 1.0
    assert b_dry._fire.wind_speed == 6.0
    import pytest
    with pytest.raises(ValueError, match="not in .* options"):
        build_samples("forest_fire", seeds=[1], weather="hurricane")


def test_unknown_dynamics_key_fails_loudly():
    import pytest

    from dmb.scenarios import build_samples

    meta = dict(build_samples("forest_fire", seeds=[1])[0].metadata)
    meta["dynamics"] = {**meta["dynamics"], "fier": {"ignition": []}}
    with pytest.raises(ValueError, match="Unknown dynamics blocks"):
        MockBridge().reset(meta)


def test_terrain_weather_diverges_slip_step():
    import pytest

    cfg = {
        "candidates": [{"x": 5, "y": 8, "slope": 1.5}],
        "sat_rate": 0.05,
        "rain_multiplier": 2.0,
        "raining": True,
        "threshold": 1.0,
        "debris_dir": [0, 1],
        "debris_len": 3,
        "weather_modifiers": {"rain-heavy": {"raining": True}, "overcast": {"raining": False}},
    }
    wet = SlipModel({**cfg, "weather_active": "rain-heavy"}, 10, 12)
    dry = SlipModel({**cfg, "weather_active": "overcast"}, 10, 12)
    for step in range(1, 61):
        wet.step(step)
        dry.step(step)
    assert wet.slip_step == 11
    assert dry.slip_step == 20
    with pytest.raises(ValueError, match="Unknown terrain weather"):
        SlipModel({**cfg, "weather_active": "snow"}, 10, 12)


def test_unknown_mobility_fails_loudly():
    import pytest

    from dmb.scenarios import build_samples

    meta = dict(build_samples("forest_fire", seeds=[1])[0].metadata)
    meta["victim_placements"] = [
        {**meta["victim_placements"][0], "mobility": "flying"}
    ]
    with pytest.raises(ValueError, match="Unknown victim mobility"):
        MockBridge().reset(meta)
