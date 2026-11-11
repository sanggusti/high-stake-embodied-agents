"""Modal app pins (#74): fallback chains, reward mirror, CLI wiring."""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_app():
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
