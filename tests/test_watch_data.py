"""L2 recorder, replay, seed splits, SFT conversion."""

import json
from types import SimpleNamespace

import pytest

from dmb.data.export import messages_to_sft_pairs
from dmb.data.splits import TEST_SEEDS, TRAIN_SEEDS, split_for_seed
from dmb.watch.recorder import JsonlWatcher, record_score
from dmb.watch.replay import narrate


def test_recorder_writes_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DMB_WATCH_DIR", str(tmp_path))
    watch = JsonlWatcher("s-1", tmp_path)
    watch.on_start({"scenario": "forest_fire", "seed": 1, "world_hash": "ab", "backend": "mock"})
    watch.on_event("action", 5.0, 1, tool="goto_location", args={"x": 1, "y": 2}, obs="Moved")
    watch.on_end({"state": {"sim_time": 5.0, "steps": 1, "victims": [], "safety_violations": 0, "coverage": 0.1}})
    record_score("s-1", {"composite": 0.5})
    events = [json.loads(line) for line in (tmp_path / "s-1.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["note", "action", "state", "note", "score"]
    assert all(e["t"] >= 0 or e["kind"] == "score" for e in events)
    assert events[1]["tool"] == "goto_location"


def test_replay_narrates(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"t": 0.0, "step": 0, "kind": "note", "event": "episode_start",'
        ' "scenario": "forest_fire", "seed": 1, "backend": "mock"}\n'
        '{"t": 5.0, "step": 1, "kind": "action", "tool": "arm", "args": {}, "obs": "Armed"}\n'
        '{"t": -1.0, "step": -1, "kind": "score", "composite": 0.5,'
        ' "rescue_recall": 0.5, "containment": 0.5, "efficiency": 0.5, "safety": 1.0}\n'
    )
    lines = narrate(path)
    assert len(lines) == 3
    assert "episode_start" in lines[0] and "forest_fire" in lines[0]
    assert "arm(" in lines[1]
    assert "composite=0.500" in lines[2]
    assert len(narrate(path, {"action"})) == 1


def test_splits_disjoint_and_guarded():
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS)
    assert split_for_seed(TRAIN_SEEDS[0]) == "train"
    assert split_for_seed(TEST_SEEDS[0]) == "test"
    with pytest.raises(ValueError):
        split_for_seed(9999)


def test_sft_pairs_one_per_tool_turn():
    user = SimpleNamespace(role="user", content="brief")
    turn1 = SimpleNamespace(
        role="assistant",
        content="I will arm first.",
        tool_calls=[SimpleNamespace(id="c1", function="arm", arguments={})],
    )
    tool = SimpleNamespace(role="tool", tool_call_id="c1", content="Armed")
    turn2 = SimpleNamespace(
        role="assistant",
        content="Now take off.",
        tool_calls=[SimpleNamespace(id="c2", function="takeoff", arguments={"altitude_m": 10})],
    )
    pairs = messages_to_sft_pairs([user, turn1, tool, turn2])
    assert len(pairs) == 2
    assert pairs[0][-1]["tool_calls"][0]["function"]["name"] == "arm"
    assert pairs[0][-1]["content"] == "I will arm first."
    assert pairs[1][-1]["tool_calls"][0]["function"]["name"] == "takeoff"
    assert len(pairs[1]) == 4  # prefix grows: user, turn1, tool, turn2


def test_sft_pairs_skip_text_only_turns():
    user = SimpleNamespace(role="user", content="brief")
    plain = SimpleNamespace(role="assistant", content="thinking", tool_calls=None)
    assert messages_to_sft_pairs([user, plain]) == []


def test_image_blocks_become_marker():
    img = SimpleNamespace(image="data:...", detail="low")
    tool = SimpleNamespace(role="tool", tool_call_id="c1", content=[img])
    user = SimpleNamespace(role="user", content="brief")
    turn = SimpleNamespace(
        role="assistant", content="ok",
        tool_calls=[SimpleNamespace(id="c2", function="land", arguments={})],
    )
    pairs = messages_to_sft_pairs([user, tool, turn])
    assert pairs[0][1]["content"] == "[frame omitted]"


def test_render_produces_gif_and_png(tmp_path):
    import json
    from dmb.scenarios import build_samples
    from dmb.watch.render import render

    meta = build_samples("forest_fire", seeds=[1])[0].metadata
    events = [
        {"t": 0.0, "step": 0, "kind": "note", "event": "episode_start",
         "scenario": "forest_fire", "seed": 1, "world_hash": meta["world_hash"]},
        {"t": 0.0, "step": 0, "kind": "action", "tool": "arm", "args": {},
         "obs": "Armed.", "pos": [0, 0]},
        {"t": 5.0, "step": 1, "kind": "edge", "trigger": "novelty",
         "summary": "found one", "detections": [{"label": "human", "x": 5, "y": 5}]},
        {"t": 5.0, "step": 1, "kind": "action", "tool": "goto_location",
         "args": {"x": 5, "y": 5}, "obs": "Moved.", "pos": [5, 5]},
        {"t": -1.0, "step": -1, "kind": "score", "composite": 0.9},
    ]
    src = tmp_path / "ep.jsonl"
    src.write_text("\n".join(json.dumps(e) for e in events))
    out = render(src, tmp_path / "replays")
    assert out["frames"] == 3  # 2 actions + done card
    gif = tmp_path / "replays" / "ep.gif"
    png = tmp_path / "replays" / "ep_final.png"
    assert gif.stat().st_size > 0 and png.stat().st_size > 0
    from PIL import Image
    assert Image.open(gif).n_frames == 3


def test_render_rejects_hash_mismatch(tmp_path):
    import json
    import pytest
    from dmb.watch.render import render

    src = tmp_path / "ep.jsonl"
    src.write_text(json.dumps(
        {"t": 0.0, "step": 0, "kind": "note", "event": "episode_start",
         "scenario": "forest_fire", "seed": 1, "world_hash": "deadbeef"}) + "\n")
    with pytest.raises(ValueError, match="world_hash mismatch"):
        render(src, tmp_path / "replays")


def test_l3_paths_link_beside_trajectories(tmp_path, monkeypatch):
    from dmb.watch.l3 import l3_enabled, l3_sample_dir

    monkeypatch.setenv("DMB_WATCH_DIR", str(tmp_path / "runs" / "r1" / "trajectories"))
    monkeypatch.delenv("DMB_L3", raising=False)
    assert l3_enabled()
    path = l3_sample_dir("s-1")
    assert path == tmp_path / "runs" / "r1" / "sim" / "s-1"
    assert path.exists()
    monkeypatch.setenv("DMB_L3", "off")
    assert not l3_enabled()


def test_compose_renders_all_layers(tmp_path):
    from types import SimpleNamespace

    from dmb.watch.compose import collect_artifacts, render_html, transcript_turns

    user = SimpleNamespace(role="user", content="brief")
    turn = SimpleNamespace(
        role="assistant", content="arming now",
        tool_calls=[SimpleNamespace(function="arm", arguments={})],
    )
    tool = SimpleNamespace(role="tool", tool_call_id="c1", content="Armed")
    turns = transcript_turns([user, turn, tool])
    assert [t["role"] for t in turns] == ["user", "assistant", "tool"]
    assert turns[1]["tool_calls"][0]["name"] == "arm"

    l3 = tmp_path / "s-1"
    l3.mkdir(parents=True)
    (l3 / "final.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    (l3 / "notes.txt").write_text("ignored")
    arts = collect_artifacts(l3)
    assert len(arts) == 1 and arts[0]["data_uri"].startswith("data:image/png;base64,")

    page = render_html(
        "Replay: s-1", turns, ["[t=0] action arm()"], {"composite": 0.5}, arts
    )
    assert "composite=0.500" in page and "arm(" in page and "data:image/png" in page
