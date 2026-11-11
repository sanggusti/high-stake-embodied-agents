"""Modal app pins (#74): fallback chains, reward mirror, CLI wiring."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location(
        "aya_app", ROOT / "infra" / "modal" / "aya_expanse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_strict_single_gpu_class():
    src = (ROOT / "infra" / "modal" / "aya_expanse.py").read_text()
    strict = re.findall(r'gpu="(?:A100|A10G|L4|H100|H200|L40S)[^"]*"', src)
    assert strict == []
    for chain in ("TRAIN_GPU", "DUAL_GPU", "INFER_GPU"):
        assert f"gpu={chain}" in src


def test_fallback_chains_shape():
    mod = _load_app()
    for name in ("TRAIN_GPU", "DUAL_GPU", "INFER_GPU"):
        chain = getattr(mod, name)
        assert isinstance(chain, list) and len(chain) >= 2
    assert "A100-80GB" in mod.DUAL_GPU  # 80GB for dual-model legs
    assert mod.INFER_GPU[0] == "A10G"  # cheapest first for inference


def test_gold_priority_mirrors_package():
    from hsea.rlvr.rewards import priority_of

    mod = _load_app()
    cases = [
        "PRIORITY 1 \u2014 IMMEDIATE DISPATCH. Task unit 7 now.",
        "PRIORITY 2 \u2014 ROUTINE. Queue for next unit.",
        "set P3 for this one",
        "no priority here",
        "",
    ]
    for text in cases:
        assert mod._gold_priority(text) == priority_of(text)


def test_inline_reward_mirrors_package():
    from hsea.rlvr.rewards import rule_reward_dispatch

    mod = _load_app()
    cases = [
        ("P1: task the nearest unit now.", "P1"),
        ("P2 noted for the record.", "P2"),
        ("P3: notify before acting, revisit.", "P3"),
        ("done.\nDECISION: comply", "P1"),
        ("", "P1"),
    ]
    for comp, gold in cases:
        assert mod._dispatch_reward([comp], [gold])[0] == \
            rule_reward_dispatch(comp, gold)


def test_stage_cmds_wired():
    src = (ROOT / "infra" / "modal" / "aya_expanse.py").read_text()
    for cmd in ("repair", "kto", "ipo", "grpo", "rft"):
        assert f'def {cmd}(' in src
        assert f'cmd == "{cmd}"' in src
    assert "prompts_path" in src
    assert 'loss_type="ipo"' in src
    assert "GRPOTrainer" in src and "KTOTrainer" in src
    assert "def rft_train(" in src
    assert "rft_train.remote()" in src


def test_rft_grpo_use_composed_schema():
    src = (ROOT / "infra" / "modal" / "aya_expanse.py").read_text()
    assert "def _gold_priority(" in src
    assert 'rec["situation"]' not in src
    assert "rec['situation']" not in src
    assert 'rec["priority"]' not in src
    assert "rec['priority']" not in src
    assert 'rec.get("priority"' not in src
    assert '"flywheel-v2"' in src
    assert "num_return_sequences=n_samples" not in src  # OOM on A10G
    assert "num_return_sequences=1" in src  # sequential RFT sampling
    assert "max_prompt_length" not in src  # removed from GRPOConfig


def test_entrypoint_per_command_defaults():
    src = (ROOT / "infra" / "modal" / "aya_expanse.py").read_text()
    assert "epochs=epochs if epochs > 0 else 3" in src  # sft
    assert "epochs=epochs if epochs > 0 else 1" in src  # repair/kto
    assert "max_steps if max_steps > 0 else 100" in src  # grpo
