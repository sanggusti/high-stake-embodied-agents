"""Tests for pointer-serve resolution + DPO-half splitter (no GPU)."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue8_dpo_half as half_mod


def _load_serve():
    class _Chain:
        def __getattr__(self, _name):
            return lambda *a, **k: _Chain()

    def _deco(*a, **k):
        def wrap(fn):
            return fn
        return wrap

    stub = types.ModuleType("modal")
    stub.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())
    stub.concurrent = _deco
    stub.web_server = _deco

    class _App:
        def __init__(self, *a, **k):
            pass

        def function(self, *a, **k):
            return _deco()

    stub.App = _App
    sys.modules["modal"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "vllm_qwen38_ft", ROOT / "infra" / "modal" / "vllm_qwen38_ft.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["modal"]
    return mod


serve = _load_serve()


def test_resolve_defaults_to_sft_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "POINTER_PATH", str(tmp_path / "missing"))
    assert serve.resolve_model_path() == serve.DEFAULT_MODEL_PATH


def test_resolve_reads_pointer(tmp_path, monkeypatch):
    pointer = tmp_path / "ACTIVE_CHECKPOINT"
    pointer.write_text("/models/qwen38-flywheel-v2-dpo-merged\n")
    monkeypatch.setattr(serve, "POINTER_PATH", str(pointer))
    assert serve.resolve_model_path() == \
        "/models/qwen38-flywheel-v2-dpo-merged"


def test_resolve_blank_pointer_falls_back(tmp_path, monkeypatch):
    pointer = tmp_path / "ACTIVE_CHECKPOINT"
    pointer.write_text("  \n")
    monkeypatch.setattr(serve, "POINTER_PATH", str(pointer))
    assert serve.resolve_model_path() == serve.DEFAULT_MODEL_PATH


def test_dpo_half_stratified_and_deterministic():
    first = half_mod.build_half()
    second = half_mod.build_half()
    assert first == second
    assert len(first) == 3499  # floor-halves: 614 + 1667 + 1218
    from collections import Counter

    counts = Counter(r["priority"] for r in first)
    assert counts == {"P1": 614, "P2": 1667, "P3": 1218}
    assert len({json_key(r) for r in first}) == 3499


def json_key(rec):
    return (rec["situation"], rec["chosen"], rec["rejected"])
