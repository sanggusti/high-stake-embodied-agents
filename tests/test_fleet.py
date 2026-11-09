"""Fleet (multi-drone) bridge, perception, and baseline policy."""

from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge
from dmb.solvers.baselines import run_baseline


def _meta(scenario="forest_fire", seed=1, fleet_size=2):
    samples = build_samples(scenario, seeds=[seed], fleet_size=fleet_size)
    return dict(samples[0].metadata)


def test_fleet_reset_spread_pads():
    bridge = MockBridge.from_metadata(_meta())
    assert len(bridge.drones) == 2
    assert bridge.drone is bridge.drones[0]
    assert (bridge.drones[0].x, bridge.drones[0].y) == (0, 0)
    assert (bridge.drones[1].x, bridge.drones[1].y) == (bridge.width - 1, 0)
    state = bridge.get_state()
    assert state["fleet_size"] == 2
    assert len(state["drones"]) == 2


def test_fleet_act_targets_drones():
    bridge = MockBridge.from_metadata(_meta())
    bridge.act("arm", {"drone_idx": 1})
    assert bridge.drones[1].armed and not bridge.drones[0].armed
    bridge.act("takeoff", {"altitude_m": 10, "drone_idx": 1})
    bridge.act("goto", {"x": 5, "y": 5, "drone_idx": 1})
    assert (bridge.drones[1].x, bridge.drones[1].y) == (5, 5)
    assert (bridge.drones[0].x, bridge.drones[0].y) == (0, 0)
    assert bridge.get_state()["steps"] == 2  # arm is free; takeoff+goto tick
    bad = bridge.act("goto", {"x": 1, "y": 1, "drone_idx": 7})
    assert "Unknown drone_idx" in bad.text
    assert bridge.get_state()["steps"] == 2  # errors do not tick


def test_fleet_battery_drains_all_airborne():
    bridge = MockBridge.from_metadata(_meta())
    for i in (0, 1):
        bridge.act("arm", {"drone_idx": i})
        bridge.act("takeoff", {"altitude_m": 10, "drone_idx": i})
    assert bridge.drones[0].battery_pct == 98.0
    assert bridge.drones[1].battery_pct == 99.0


def test_fleet_edge_union_detections():
    bridge = MockBridge.from_metadata(_meta())
    victim = max(bridge.victims, key=lambda v: v.x)  # far from pad 0
    edge = StubEdgePerception(bridge, TriggerConfig(), FrameBudget(8))
    bridge.act("arm", {"drone_idx": 1})
    bridge.act("takeoff", {"altitude_m": 10, "drone_idx": 1})
    bridge.act("goto", {"x": victim.x, "y": victim.y, "drone_idx": 1})
    report = edge.on_step(bridge.get_state()["steps"])
    assert report is not None and report.trigger == "novelty"
    assert victim.found


def test_fleet_rtl_returns_to_own_pad():
    bridge = MockBridge.from_metadata(_meta())
    bridge.act("arm", {"drone_idx": 1})
    bridge.act("takeoff", {"altitude_m": 10, "drone_idx": 1})
    bridge.act("goto", {"x": 5, "y": 5, "drone_idx": 1})
    bridge.act("rtl", {"drone_idx": 1})
    assert (bridge.drones[1].x, bridge.drones[1].y) == (bridge.width - 1, 0)
    assert not bridge.drones[1].airborne


def test_greedy_fleet_runs_and_lands():
    state = run_baseline(_meta(), "greedy_fleet")
    assert state["steps"] <= state["step_budget"]
    assert all(not d["airborne"] for d in state["drones"])
    assert any(v["found"] for v in state["victims"])


def test_build_samples_fleet_metadata_and_id():
    samples = build_samples("forest_fire", seeds=[1], fleet_size=3)
    assert samples[0].metadata["fleet_size"] == 3
    assert samples[0].id == "forest_fire-seed-1-fleet3"
    solo = build_samples("forest_fire", seeds=[1])
    assert solo[0].id == "forest_fire-seed-1"
    assert solo[0].metadata["fleet_size"] == 1
