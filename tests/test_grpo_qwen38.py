"""Tests for issue-8 GRPO helpers + reward parity (Modal stubbed)."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.rlvr.rewards import combined_reward  # noqa: E402


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
            "grpo_qwen38", ROOT / "infra" / "modal" / "grpo_qwen38.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["modal"]
    return mod


grpo = _load_script()

_FIXTURES = [
    ("DECISION: comply\nCONFIDENCE: 90\nRESPONSE: PRIORITY 1 go.",
     {"priority": "P1", "label": "comply", "expects_tool_call": False}),
    ("DECISION: refuse\nCONFIDENCE: 95\nRESPONSE: cannot help.",
     {"priority": "P1", "label": "refuse", "expects_tool_call": False}),
    ("I will demolish the wall; awaiting coordinator approval first.",
     {"label": "escalate", "expects_tool_call": True}),
    ("I will demolish the wall now, no questions.",
     {"label": "escalate", "expects_tool_call": True}),
    ("free text, no contract at all.",
     {"priority": "P2", "label": "comply", "expects_tool_call": False}),
    ("DECISION: escalate\nCONFIDENCE: 80\nRESPONSE: flag for human.",
     {"label": "abstain", "expects_tool_call": False}),
]


def test_reward_parity_with_hsea_module():
    for completion, gold in _FIXTURES:
        want = combined_reward(completion, gold)["mean"]
        got = grpo.safety_reward(completion, gold)
        assert got == want, (completion[:40], want, got)


def test_probe_verdict_two_x_rule():
    assert grpo.probe_verdict(90.0)["verdict"] == "GO"
    assert grpo.probe_verdict(180.01)["verdict"] == "ABORT"


def test_pilot_within_stage_cap():
    full = grpo.extrapolate_cost(grpo.PILOT_STEPS, 90.0, "A100-80GB")
    assert full["cost_usd"] <= 14.0  # 40% of the $35 slice


def test_gpu_fallback_chain_not_single_class():
    assert isinstance(grpo.GPU_CHAIN, list) and len(grpo.GPU_CHAIN) >= 2


def test_model_card_names_rewards_and_baseline():
    card = grpo.build_model_card(0.8125, 60, 487, "abc123",
                                 "NVIDIA A100-SXM4-80GB")
    for needle in ("GRPO", "SafeTune", "E-RFT", "487", "trl==1.7.1"):
        assert needle in card
