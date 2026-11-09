from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.sim.bridge import MockBridge


def _bridge():
    return MockBridge.from_metadata(
        {
            "scenario": "forest_fire",
            "seed": 1,
            "grid": {"width": 10, "height": 10},
            "victim_placements": [{"category": "human", "x": 1, "y": 0}],
            "hazards": [],
        }
    )


def test_novelty_report_fires_once_per_victim():
    bridge = _bridge()
    edge = StubEdgePerception(bridge, TriggerConfig(), FrameBudget(8))
    bridge.act("arm", {})
    assert edge.on_step(0) is None  # grounded: no detections
    bridge.act("takeoff", {"altitude_m": 10})
    report = edge.on_step(1)
    assert report is not None and report.trigger == "novelty"
    assert bridge.get_state()["victims"][0]["found"] is True
    # second scan: already known, and heartbeat interval not reached
    assert edge.on_step(2) is None


def test_snapshot_budget_is_enforced():
    bridge = _bridge()
    edge = StubEdgePerception(bridge, TriggerConfig(), FrameBudget(1))
    assert edge.on_demand(0).frame_ref is not None
    assert edge.on_demand(1).frame_ref is None


def test_mobility_detectability_profile():
    from dmb.perception.edge import (
        MOBILITY_CONFIDENCE,
        MOBILITY_RADIUS_FACTOR,
        FrameBudget,
        StubEdgePerception,
        TriggerConfig,
    )
    from dmb.sim.bridge import MockBridge

    assert MOBILITY_RADIUS_FACTOR["buried"] < MOBILITY_RADIUS_FACTOR["trapped"] < 1.0
    assert MOBILITY_CONFIDENCE["buried"] < MOBILITY_CONFIDENCE["trapped"]

    bridge = MockBridge()
    bridge.reset(
        {
            "scenario": "mock",
            "seed": 0,
            "grid": {"width": 10, "height": 10},
            "victim_placements": [
                {"category": "human", "x": 2, "y": 0, "mobility": "static"},
                {"category": "human", "x": 2, "y": 1, "mobility": "buried"},
            ],
            "hazards": [],
        }
    )
    bridge.drone.airborne = True
    edge = StubEdgePerception(bridge, TriggerConfig(), FrameBudget(max_snapshots=8))
    detections, _ = edge._scan()
    # Static victim at dist 2.0 seen; buried at dist ~2.2 beyond radius/3.
    assert [(d.x, d.y) for d in detections] == [(2, 0)]
    bridge.drone.x, bridge.drone.y = 2, 1
    detections, _ = edge._scan()
    buried = [d for d in detections if (d.x, d.y) == (2, 1)][0]
    assert buried.confidence == MOBILITY_CONFIDENCE["buried"]


def test_time_of_day_shrinks_radius():
    import pytest

    from dmb.perception.edge import (
        FrameBudget,
        StubEdgePerception,
        TriggerConfig,
    )
    from dmb.sim.bridge import MockBridge

    def scan_at(tod):
        bridge = MockBridge()
        bridge.reset(
            {
                "scenario": "mock",
                "seed": 0,
                "grid": {"width": 10, "height": 10},
                "victim_placements": [
                    {"category": "human", "x": 2, "y": 0, "mobility": "static"}
                ],
                "hazards": [],
            }
        )
        bridge.drone.airborne = True
        edge = StubEdgePerception(
            bridge, TriggerConfig(time_of_day=tod), FrameBudget(max_snapshots=8)
        )
        return edge._scan()[0]

    assert len(scan_at("day")) == 1
    assert len(scan_at("night")) == 0  # dist 2.0 > 3.0 * 0.5
    with pytest.raises(ValueError, match="Unknown time_of_day"):
        scan_at("eclipse")
