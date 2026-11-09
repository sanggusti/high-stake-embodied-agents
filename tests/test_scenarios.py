from dmb.scenarios import build_samples, list_scenarios, load_scenario, world_hash


def test_all_scenarios_load():
    assert sorted(list_scenarios()) == [
        "earthquake",
        "earthquake_hard",
        "firms_peat_kalimantan_01",
        "firms_peat_sumatra_01",
        "firms_peat_sumatra_02",
        "flash_flood",
        "flash_flood_hard",
        "flood",
        "flood_hard",
        "forest_fire",
        "forest_fire_large",
        "forest_fire_peat",
        "landslide",
        "landslide_hard",
        "landslide_large",
        "tsunami",
        "tsunami_hard",
    ]
    for name in list_scenarios():
        config = load_scenario(name)
        assert config.grid["width"] > 0
    assert load_scenario("forest_fire_large").raw["variant_of"] == "forest_fire"
    assert load_scenario("forest_fire_peat").raw["variant_of"] == "forest_fire"
    assert load_scenario("landslide_large").raw["variant_of"] == "landslide"


def test_samples_are_deterministic():
    first = build_samples("forest_fire", seeds=[7])
    second = build_samples("forest_fire", seeds=[7])
    assert first[0].metadata == second[0].metadata
    assert first[0].metadata["world_hash"] == world_hash(
        "forest_fire", 7, first[0].metadata["victim_placements"]
    )


def test_distinct_seeds_differ():
    a = build_samples("forest_fire", seeds=[1])[0].metadata["world_hash"]
    b = build_samples("forest_fire", seeds=[2])[0].metadata["world_hash"]
    assert a != b


def test_brief_states_grid_bounds():
    from dmb.scenarios import build_samples

    sample = build_samples("forest_fire", seeds=[1])[0]
    assert "10x10" in sample.input
    assert "x=0..9, y=0..9" in sample.input
    assert "(0, 0)" in sample.input


def test_time_of_day_axis():
    import pytest

    from dmb.scenarios import build_samples

    sample = build_samples("flood", seeds=[2], time_of_day="night")[0]
    assert sample.id == "flood-seed-2-night"
    assert sample.metadata["time_of_day_active"] == "night"
    both = build_samples("forest_fire", seeds=[1], weather="clear", time_of_day="dusk")[0]
    assert both.id == "forest_fire-seed-1-clear-dusk"
    with pytest.raises(ValueError, match="time_of_day"):
        build_samples("flood", seeds=[2], time_of_day="eclipse")


def test_bands_cover_easy_medium_hard():
    from dmb.scenarios import build_samples, list_scenarios

    bands = set()
    for scenario in list_scenarios():
        for sample in build_samples(scenario, seeds=list(range(1, 31))):
            bands.add(sample.metadata["difficulty_score"]["band"])
    assert bands == {"easy", "medium", "hard"}


def test_large_variants_state_bounds_and_placements():
    for name in ("forest_fire_large", "landslide_large"):
        sample = build_samples(name, seeds=[1])[0]
        assert "20x20" in sample.input
        assert "x=0..19, y=0..19" in sample.input
        md = sample.metadata
        assert md["grid"] == {"width": 20, "height": 20}
        for v in md["victim_placements"]:
            assert 0 <= v["x"] < 20 and 0 <= v["y"] < 20
        for hz in md["hazards"]:
            assert 0 <= hz["x"] < 20 and 0 <= hz["y"] < 20
        again = build_samples(name, seeds=[1])[0]
        assert again.metadata == md
        assert md["world_hash"] == world_hash(name, 1, md["victim_placements"])


def test_pack_excludes_large_variants():
    from evals.scenario_pack import pack_scenarios, scenario_pack

    assert pack_scenarios() == [
        "earthquake",
        "flash_flood",
        "flood",
        "forest_fire",
        "landslide",
        "tsunami",
    ]
    ids = [s.id for s in scenario_pack().dataset]
    assert len(ids) == 18
    assert not [i for i in ids if "large" in i]


def test_baseline_pack_matches_lm_pack():
    from evals.pack_baselines import pack_heuristic, pack_random
    from evals.scenario_pack import scenario_pack

    want = sorted(s.id for s in scenario_pack().dataset)
    assert sorted(s.id for s in pack_heuristic().dataset) == want
    assert sorted(s.id for s in pack_random().dataset) == want


def test_large_grid_mock_flight():
    from dmb.sim.bridge import MockBridge

    md = dict(build_samples("forest_fire_large", seeds=[1])[0].metadata)
    bridge = MockBridge.from_metadata(md)
    assert (bridge.width, bridge.height) == (20, 20)
    assert "Armed" in bridge.act("arm", {}).text
    assert bridge.act("takeoff", {"altitude_m": 10}).text.startswith("Takeoff")
    assert "(19, 19)" in bridge.act("goto", {"x": 19, "y": 19}).text
    assert "Landed" in bridge.act("land", {}).text
    bridge.close()
