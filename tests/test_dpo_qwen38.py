"""Tests for issue-8 DPO prep helpers (Modal stubbed, no GPU)."""

import importlib.util
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
            "dpo_qwen38", ROOT / "infra" / "modal" / "dpo_qwen38.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["modal"]
    return mod


dpo = _load_script()


def _row(chosen="PRIORITY 1 dispatch now.", rejected="PRIORITY 3 monitor."):
    return {"situation": "Crisis report: injuries.", "priority": "P1",
            "chosen": chosen, "rejected": rejected, "split": "train"}


def test_coerce_pair_chat_shape():
    pair = dpo.coerce_pair(_row())
    assert pair["prompt"] == [{"role": "user",
                               "content": "Crisis report: injuries."}]
    assert pair["chosen"][-1] == {"role": "assistant",
                                  "content": "PRIORITY 1 dispatch now."}
    assert pair["rejected"][-1]["role"] == "assistant"
    assert pair["priority"] == "P1"


def test_coerce_pair_rejects_malformed():
    with pytest.raises(ValueError):
        dpo.coerce_pair({"situation": "x"})
    with pytest.raises(ValueError):
        dpo.coerce_pair(_row(chosen="  "))
    with pytest.raises(ValueError):
        dpo.coerce_pair("not-a-dict")


def test_debias_filter_drops_extreme_ratio_only():
    assert dpo.coerce_pair(_row()) is not None
    assert dpo.coerce_pair(_row(chosen="P1 " + "x" * 1000,
                                rejected="P3.")) is None
    assert dpo.coerce_pair(_row(chosen="P1.",
                                rejected="P3 " + "y" * 1000)) is None


def test_debias_filter_drops_zero_v2_rows():
    import json

    dropped = 0
    n = 0
    for line in open(ROOT / "data" / "flywheel-v2" / "dpo.jsonl"):
        if line.strip():
            n += 1
            if dpo.coerce_pair(json.loads(line)) is None:
                dropped += 1
    assert n == 7000
    assert dropped == 0


def test_count_steps_dpo_arm():
    assert dpo.count_steps(7000, 2, 4, 1) == 875


def test_probe_verdict_two_x_rule():
    assert dpo.probe_verdict(13.0)["verdict"] == "GO"
    assert dpo.probe_verdict(26.01)["verdict"] == "ABORT"


def test_gpu_fallback_chain_not_single_class():
    assert isinstance(dpo.GPU_CHAIN, list) and len(dpo.GPU_CHAIN) >= 2


def test_model_card_names_loss_and_parent():
    card = dpo.build_model_card("sigmoid_norm", 0.6931, 875, 7000, 0,
                                "abc123", "NVIDIA A100-SXM4-80GB")
    for needle in ("sigmoid_norm", "stage-1 SFT adapter", "7,000",
                   "trl==1.7.1", "abc123"):
        assert needle in card


def test_classify_gpu_distinguishes_h200():
    assert dpo.classify_gpu("NVIDIA H200") == "H200"
    assert dpo.classify_gpu("NVIDIA H100 80GB HBM3") == "H100"
    assert dpo.classify_gpu("NVIDIA A100-SXM4-80GB") == "A100-80GB"
    assert dpo.classify_gpu("mystery") == "A100-80GB"


def test_probe_verdict_cost_gate():
    assert dpo.probe_verdict(7.0, 5.0, 12.0)["verdict"] == "GO"
    assert dpo.probe_verdict(7.0, 12.01, 12.0)["verdict"] == "NO-GO"
    assert dpo.probe_verdict(999.0, 1.0, 12.0)["verdict"] == "ABORT"
