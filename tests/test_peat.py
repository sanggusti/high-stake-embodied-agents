"""Peatland fire dynamics (#50): smolder, breakout, drought, haze."""

from dmb.physics.fire import FireModel
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge

# Small exact-pin config: eastern block (x >= 2) over deep peat.
SMALL_PEAT = {
    "ignition": [{"x": 2, "y": 2}],
    "moisture": 0.15,
    "wind_dir": [1, 0],
    "wind_speed": 4.0,
    "fuel_default": 0.5,
    "burn_duration": 25,
    "ignition_threshold": 1.0,
    "base_ros": 0.06,
    "peat": {
        "depth_default": 0.0,
        "peat_blocks": [{"x0": 2, "y0": 0, "x1": 5, "y1": 5, "depth": 0.8}],
        "smolder_ros": 0.07,
        "smolder_threshold": 0.5,
        "breakout_threshold": 1.75,
        "drought": 1.5,
        "smolder_duration_mult": 3.0,
        "haze_per_burning": 0.05,
        "haze_per_smoldering": 0.08,
    },
}


def _run(config, width=6, height=6, steps=60):
    model = FireModel(config, width, height)
    for step in range(1, steps + 1):
        model.step(step)
    return model


def test_peat_snapshot_smolder_then_breakout():
    model = _run(SMALL_PEAT)
    snap = model.snapshot()
    assert snap["smolder"] == [
        "2,0@17", "2,1@11", "2,2@0", "2,3@11", "2,4@17", "2,5@23",
        "3,0@17", "3,1@11", "3,2@11", "3,3@11", "3,4@17", "3,5@22",
        "4,0@19", "4,1@16", "4,2@15", "4,3@16", "4,4@19", "4,5@23",
        "5,0@23", "5,1@21", "5,2@20", "5,3@21", "5,4@23", "5,5@26",
    ]
    assert snap["ignited"] == [
        "1,0@58", "1,1@51", "1,2@50", "1,3@51", "1,4@56",
        "2,0@41", "2,1@35", "2,2@25", "2,3@35", "2,4@41", "2,5@47",
        "3,0@41", "3,1@35", "3,2@35", "3,3@35", "3,4@41", "3,5@46",
        "4,0@43", "4,1@40", "4,2@39", "4,3@40", "4,4@43", "4,5@47",
        "5,0@47", "5,1@45", "5,2@44", "5,3@45", "5,4@47", "5,5@50",
    ]
    # Smolder precedes breakout on every peat cell.
    for cell, start in model.ignited.items():
        if cell in model.smolder:
            assert model.smolder[cell] < start


def test_no_peat_bit_identical_to_legacy():
    legacy_cfg = {
        "ignition": [{"x": 2, "y": 2}],
        "moisture": 0.3,
        "wind_dir": [1, 0],
        "wind_speed": 3.0,
        "fuel_default": 0.7,
        "burn_duration": 25,
        "ignition_threshold": 1.0,
        "base_ros": 0.06,
    }
    model = _run(legacy_cfg, steps=30)
    assert model.snapshot()["ignited"] == ["2,2@0", "3,1@27", "3,2@24", "3,3@27"]
    assert model.snapshot()["smolder"] == []
    assert model.drought == 1.0


def test_drought_accelerates_surface_spread():
    calm = _run(
        {**SMALL_PEAT, "peat": {**SMALL_PEAT["peat"], "drought": 1.0}}, steps=60
    )
    dry = _run(
        {**SMALL_PEAT, "peat": {**SMALL_PEAT["peat"], "drought": 2.5}}, steps=60
    )
    assert min(dry.ignited.values()) < min(calm.ignited.values())
    assert len(dry.ignited) > len(calm.ignited)


def test_peat_weather_diverges_and_repeats():
    def run_weather(weather):
        md = dict(build_samples("forest_fire_peat", seeds=[1], weather=weather)[0].metadata)
        fire = dict(md["dynamics"]["fire"])
        fire["weather_active"] = md["weather_active"]
        return _run(fire, width=10, height=10)

    moderate = run_weather("drought-moderate")
    extreme = run_weather("drought-extreme")
    assert (len(moderate.ignited), len(moderate.smolder)) == (27, 50)
    assert (len(extreme.ignited), len(extreme.smolder)) == (57, 50)
    assert min(moderate.ignited.values()) == 35
    assert min(extreme.ignited.values()) == 19
    assert moderate.drought == 1.2
    assert extreme.drought == 1.8
    repeat = run_weather("drought-extreme")
    assert repeat.snapshot() == extreme.snapshot()


def test_haze_rises_and_caps():
    model = _run(SMALL_PEAT)
    assert model.haze_index(1) == 0.08  # one smoldering cell, no flame yet
    assert model.haze_index(1) < model.haze_index(15) <= 1.0
    assert model.haze_index(60) == 1.0
    # Peat smoke is thicker per cell than surface flame.
    assert model.haze_per_smoldering > model.haze_per_burning


def test_peat_scenario_end_to_end():
    sample = build_samples("forest_fire_peat", seeds=[1], weather="drought-extreme")[0]
    assert sample.id == "forest_fire_peat-seed-1-drought-extreme"
    md = dict(sample.metadata)
    bridge = MockBridge()
    bridge.reset(md)
    assert bridge._fire.drought == 1.8
    assert bridge._fire.peat_at((5, 5)) == 0.8
    assert bridge._fire.peat_at((0, 0)) == 0.0
    # Early episode: ignition smolders underground, not yet a flight hazard.
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    assert (5, 5) in bridge._fire.smolder
    assert (5, 5) not in bridge._fire.danger_steps()
    assert bridge._fire.haze_index(bridge.get_state()["steps"]) > 0
    again = build_samples("forest_fire_peat", seeds=[1], weather="drought-extreme")[0]
    assert again.metadata == sample.metadata
    bridge.close()
