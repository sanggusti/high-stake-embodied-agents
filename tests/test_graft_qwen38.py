"""Tests for the graft tensor resolver (pure, no GPU)."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    class _Chain:
        def __getattr__(self, _name):
            return lambda *a, **k: _Chain()

    stub = types.ModuleType("modal")
    stub.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())

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
            "graft_qwen38", ROOT / "infra" / "modal" / "graft_qwen38.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["modal"]
    return mod


graft = _load()


def test_resolve_exact_match():
    merged = {"model.layers.0.mlp.gate_proj.weight"}
    assert graft.resolve_tensor(
        merged, "model.layers.0.mlp.gate_proj.weight") == \
        "model.layers.0.mlp.gate_proj.weight"


def test_resolve_strips_language_model_prefix():
    merged = {"model.layers.0.mlp.gate_proj.weight"}
    assert graft.resolve_tensor(
        merged, "language_model.model.layers.0.mlp.gate_proj.weight") == \
        "model.layers.0.mlp.gate_proj.weight"


def test_resolve_vision_returns_none():
    merged = {"model.layers.0.mlp.gate_proj.weight"}
    assert graft.resolve_tensor(
        merged, "vision_model.patch_embed.weight") is None
    assert graft.resolve_tensor(
        merged, "language_model.model.embed_tokens.weight") is None
