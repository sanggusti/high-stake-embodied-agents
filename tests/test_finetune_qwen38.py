"""Tests for issue-8 SFT prep helpers (Modal stubbed, no GPU)."""

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
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
            "finetune_qwen38", ROOT / "infra" / "modal" / "finetune_qwen38.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["modal"]
    return mod


ft = _load_script()


def test_normalize_plain_string_passthrough():
    assert ft.normalize_content("hello") == "hello"


def test_normalize_joins_disasterm3_fragments():
    frags = ["Describe damage.", "\n", "[", "f", "r", "a", "m", "e", "]"]
    assert ft.normalize_content(frags) == "Describe damage.\n[frame]"


def test_normalize_flattens_text_blocks():
    blocks = [{"type": "text", "text": "a"}, {"text": "b"}]
    assert ft.normalize_content(blocks) == "ab"


def test_normalize_rejects_unknown_block():
    with pytest.raises(ValueError):
        ft.normalize_content([{"type": "image", "url": "x"}])


def test_normalize_messages_drops_empties_keeps_roles():
    turns = ft.normalize_messages([
        {"role": "user", "content": "  "},
        {"role": "user", "content": "q?"},
        {"role": "assistant", "content": ["y", "e", "s"]},
    ])
    assert turns == [
        {"role": "user", "content": "q?"},
        {"role": "assistant", "content": "yes"},
    ]


def test_normalize_messages_requires_assistant():
    with pytest.raises(ValueError):
        ft.normalize_messages([{"role": "user", "content": "q?"}])


def test_normalize_messages_rejects_role():
    with pytest.raises(ValueError):
        ft.normalize_messages([{"role": "king", "content": "x"}])


def test_normalize_tool_calls_openai_shape():
    calls = ft.normalize_tool_calls([
        {"function": {"name": "arm", "arguments": "{}"},
         "id": "x", "type": "function"},
        {"name": "takeoff", "arguments": {"altitude_m": 30}},
    ])
    assert calls == [
        {"name": "arm", "arguments": {}},
        {"name": "takeoff", "arguments": {"altitude_m": 30}},
    ]


def test_normalize_tool_calls_json_string_args():
    calls = ft.normalize_tool_calls(
        [{"name": "takeoff", "arguments": '{"altitude_m": 30}'}])
    assert calls == [{"name": "takeoff", "arguments": {"altitude_m": 30}}]


def test_normalize_tool_calls_rejects_bad_args():
    with pytest.raises(ValueError):
        ft.normalize_tool_calls([{"name": "x", "arguments": "[1,2]"}])
    with pytest.raises(ValueError):
        ft.normalize_tool_calls([{"name": "x", "arguments": "{oops"}])


def test_empty_assistant_with_tool_calls_kept():
    turns = ft.normalize_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "arm", "arguments": "{}"}]},
    ])
    assert turns[1] == {"role": "assistant", "content": "",
                        "tool_calls": [{"name": "arm", "arguments": {}}]}


def test_tool_calls_on_user_turn_rejected():
    with pytest.raises(ValueError):
        ft.normalize_messages([
            {"role": "user", "content": "go",
             "tool_calls": [{"name": "arm", "arguments": {}}]},
            {"role": "assistant", "content": "done"},
        ])


def test_coerce_record_keeps_source():
    rec = ft.coerce_record({"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ], "source": "ladi-v2"})
    assert rec == {"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ], "source": "ladi-v2"}


def test_lora_regex_targets_text_tower_only():
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.language_model.embed_tokens",
    ]
    rx = re.compile(ft.lora_regex_for(names))
    assert rx.search(names[0]) and rx.search(names[1])
    assert not rx.search(names[2])
    assert not rx.search(names[3])


def test_lora_regex_pure_lm_fallback():
    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"]
    rx = re.compile(ft.lora_regex_for(names))
    assert rx.search(names[0]) and rx.search(names[1])


def test_lora_regex_empty_raises():
    with pytest.raises(ValueError):
        ft.lora_regex_for(["model.embed_tokens"])


def test_model_card_pins_revs_and_config():
    card = ft.build_model_card(1.2345, 345, 1875849, True, "abc123",
                               "NVIDIA A100-SXM4-80GB", "reference impl")
    for needle in (ft.BASE_MODEL, ft.BASE_REVISION, ft.DATA_REPO,
                   ft.DATA_REVISION, "r=32", "packing", "research-only",
                   "abc123", "1.23", "transformers==5.22.2", "trl==1.7.1",
                   "A100", "reference impl"):
        assert needle in card


def test_count_steps_matches_honest_math():
    # Binding: steps = rows / eff-batch x epochs (troubleshoot note).
    assert ft.count_steps(8673, 4, 2, 3) == 3255
    assert ft.count_steps(7000, 4, 2, 1) == 875


def test_extrapolate_cost_scales_with_rate():
    full = ft.extrapolate_cost(3255, 7.0, "A100-80GB")
    assert full["hours"] == pytest.approx(3255 * 7.0 / 3600 + 0.5, rel=1e-3)
    assert full["cost_usd"] == pytest.approx(full["hours"] * 2.49, rel=1e-3)
    h100 = ft.extrapolate_cost(3255, 7.0, "H100")
    assert h100["cost_usd"] > full["cost_usd"]


def test_probe_verdict_two_x_rule():
    assert ft.probe_verdict(7.0)["verdict"] == "GO"
    assert ft.probe_verdict(14.0)["verdict"] == "GO"
    assert ft.probe_verdict(14.01)["verdict"] == "ABORT"


def test_gpu_fallback_chain_not_single_class():
    assert isinstance(ft.GPU_CHAIN, list) and len(ft.GPU_CHAIN) >= 2
    assert "A100-80GB" in ft.GPU_CHAIN


def test_classify_gpu_distinguishes_h200():
    assert ft.classify_gpu("NVIDIA H200") == "H200"
    assert ft.classify_gpu("NVIDIA H100 80GB HBM3") == "H100"
    assert ft.classify_gpu("NVIDIA A100-SXM4-80GB") == "A100-80GB"
    assert ft.classify_gpu("mystery") == "A100-80GB"


def test_probe_verdict_cost_gate():
    assert ft.probe_verdict(7.0, 5.0, 12.0)["verdict"] == "GO"
    assert ft.probe_verdict(7.0, 12.01, 12.0)["verdict"] == "NO-GO"
    assert ft.probe_verdict(999.0, 1.0, 12.0)["verdict"] == "ABORT"
