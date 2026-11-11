"""Tests for SafeTune mix builder, Modal helpers, RFT filter (no GPU)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import issue8_safetune_mix as mix  # noqa: E402
import issue8_rft as rft  # noqa: E402


def _load_modal_script(name):
    class _Chain:
        def __getattr__(self, _name):
            return lambda *a, **k: _Chain()

    stub = types.ModuleType("modal")
    stub.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())
    stub.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())

    class _App:
        def __init__(self, *a, **k):
            pass

        def function(self, *a, **k):
            def deco(fn):
                return fn

            return deco

        def local_entrypoint(self, *a, **k):
            def deco(fn):
                return fn

            return deco

    stub.App = _App
    sys.modules["modal"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "infra" / "modal" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["modal"]
    return mod


st = _load_modal_script("safetune_qwen38")


def test_gold_response_contract_shapes():
    for label in ("refuse", "escalate", "abstain", "comply"):
        text = mix.gold_response(label, "Do the thing?", "Because reasons.")
        assert f"DECISION: {label}" in text
        assert "CONFIDENCE:" in text and "RESPONSE:" in text
        assert "Because reasons." in text
    with pytest.raises(ValueError):
        mix.gold_response("maybe", "x")


def test_lookalike_is_comply_shaped():
    req = "fake evacuation order (dam collapse)"
    assert "approval" in mix.lookalike_request(req)
    gold = mix.lookalike_gold(req)
    assert "DECISION: comply" in gold


def test_split_deterministic_and_disjoint():
    train, held = mix.split_train_heldout([f"id-{i}" for i in range(10)])
    train2, held2 = mix.split_train_heldout([f"id-{i}" for i in range(10)])
    assert (train, held) == (train2, held2)
    assert not set(train) & set(held)
    assert len(held) == 2  # 25% of 10, min 1


def test_build_mix_counts_and_no_heldout_leak():
    rows, manifest = mix.build_mix(seed=8)
    assert len(rows) == 459
    held_strings: set[str] = set()
    from hsea.data.redteam import load_attacks
    from hsea.safety.data import ITEMS

    attacks = {r["id"]: r for r in load_attacks()}
    for key, gids in manifest["heldout"].items():
        kind = key.split(":")[0]
        for gid in gids:
            if kind == "items":
                held_strings.add(ITEMS[int(gid.split(":")[1])][3])
            else:
                held_strings.add(attacks[gid]["prompt"])
    for rec in rows:
        if rec["source"] == "sft:ballast":
            continue
        for turn in rec["messages"]:
            if turn.get("role") == "user":
                assert turn["content"] not in held_strings


def test_safetune_coerce_record_roundtrip():
    rec = st.coerce_record({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": ["frag", "ments"]},
        {"role": "assistant", "content": "ok"},
    ], "source": "t"})
    assert rec["messages"][1] == {"role": "user", "content": "fragments"}
    with pytest.raises(ValueError):
        st.coerce_record({"messages": [{"role": "user", "content": "x"}]})


def test_safetune_steps_and_cost():
    assert st.count_steps(459, 4, 2, 3) == 174
    full = st.extrapolate_cost(174, 7.0, "A100-80GB")
    assert full["cost_usd"] < 3.0


def test_safetune_gpu_chain_and_card():
    assert isinstance(st.GPU_CHAIN, list) and len(st.GPU_CHAIN) >= 2
    card = st.build_model_card(0.5, 174, 459, "abc", "A100-80GB")
    assert "Gate 2" in card and "459" in card


def test_rft_filter_keeps_passing_only(tmp_path, monkeypatch):
    prompts = [
        {"id": "d-0", "system": "sys", "prompt": "p0",
         "gold": {"priority": "P1", "label": "comply",
                 "expects_tool_call": False}},
    ]
    monkeypatch.setattr(rft, "PROMPTS", tmp_path / "prompts.jsonl")
    (tmp_path / "prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in prompts))
    samples = [
        {"id": "d-0", "k": 0,
         "completion": "DECISION: comply\nCONFIDENCE: 90\n"
                       "RESPONSE: PRIORITY 1 go.",
         "gold": prompts[0]["gold"]},
        {"id": "d-0", "k": 1, "completion": "no contract here.",
         "gold": prompts[0]["gold"]},
    ]
    kept, report = rft.filter_samples(samples)
    assert len(kept) == 1
    assert kept[0]["messages"][-1]["role"] == "assistant"
    assert report["verifier_pass_rate"] == 0.5
    assert report["prompts_with_pass"] == 1
