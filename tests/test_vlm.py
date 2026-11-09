"""Edge-VLM baseline: protocol parsing + trigger integration (stubbed VLM)."""

from dmb.perception.edge import Detection, FrameBudget, TriggerConfig
from dmb.perception.vlm import VLMEdgePerception, parse_detections
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge


def test_parse_detections_protocol():
    dets = parse_detections("human,3,4\nchild,1,2,0.9\nnonsense\nbird,5,5\nanimal,x,y\n")
    assert [(d.label, d.x, d.y) for d in dets] == [("human", 3, 4), ("child", 1, 2)]
    assert dets[0].confidence == 0.5
    assert dets[1].confidence == 0.9


def _oracle_vlm_factory(bridge: MockBridge):
    def completer(image_data_url: str, prompt: str) -> str:
        assert image_data_url.startswith("data:image/png;base64,")
        assert "48 pixels" in prompt
        lines = []
        for v in bridge.victims_within(3.0):
            lines.append(f"{v.category},{v.x},{v.y}")
        return "\n".join(lines)

    return completer


def test_vlm_novelty_flow_with_oracle():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    edge = VLMEdgePerception(bridge, TriggerConfig(), FrameBudget(8), _oracle_vlm_factory(bridge))
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    victim = bridge.victims[0]
    bridge.act("goto", {"x": victim.x, "y": victim.y})
    report = edge.on_step(bridge.get_state()["steps"])
    assert report is not None and report.trigger == "novelty"
    assert bridge.get_state()["victims"][0]["found"] is True


def test_vlm_mismatched_kind_does_not_mark_found():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    victim = bridge.victims[0]
    wrong = "animal" if victim.category != "animal" else "human"

    def liar(image_data_url: str, prompt: str) -> str:
        return f"{wrong},{victim.x},{victim.y}"

    edge = VLMEdgePerception(bridge, TriggerConfig(), FrameBudget(8), liar)
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    bridge.act("goto", {"x": victim.x, "y": victim.y})
    edge.on_step(bridge.get_state()["steps"])
    assert all(v["found"] is False for v in bridge.get_state()["victims"])


def test_vlm_match_victim_prefers_nearest():
    bridge = MockBridge.from_metadata({
        "scenario": "flood", "seed": 0, "grid": {"width": 14, "height": 10},
        "victim_placements": [
            {"category": "human", "x": 7, "y": 7},
            {"category": "human", "x": 7, "y": 6},
        ],
        "hazards": [], "difficulty": {}, "dynamics": {},
    })
    edge = VLMEdgePerception(
        bridge, TriggerConfig(), FrameBudget(8), lambda url, prompt: "")
    assert edge._match_victim(Detection("human", 7, 6, 0.9)).y == 6
    assert edge._match_victim(Detection("human", 7, 7, 0.9)).y == 7


def test_vlm_failure_degrades_quietly():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))

    def boom(image_data_url: str, prompt: str) -> str:
        raise RuntimeError("endpoint down")

    edge = VLMEdgePerception(bridge, TriggerConfig(), FrameBudget(8), boom)
    assert edge.on_step(0) is None
    report = edge.on_demand(0)
    assert report.detections == []
