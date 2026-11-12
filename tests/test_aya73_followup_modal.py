"""Modal app pins (#73 follow-up): fallback chains, mirrors, wiring."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "infra" / "modal" / "finetune_aya73_followup.py"


def _const(path: Path, name: str):
    """Module-level constant via AST (no modal import needed)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _func(path: Path, name: str):
    """Top-level pure function via AST exec (no modal import needed)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns: dict = {}
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {path}")


def test_no_strict_single_gpu_class():
    src = APP.read_text()
    strict = re.findall(r'gpu="(?:A100|A10G|L4|H100|H200|L40S)[^"]*"', src)
    assert strict == []
    for chain in ("TRAIN_GPU", "DUAL_GPU", "INFER_GPU"):
        assert f"gpu={chain}" in src


def test_fallback_chains_shape():
    for name in ("TRAIN_GPU", "DUAL_GPU", "INFER_GPU"):
        chain = _const(APP, name)
        assert isinstance(chain, list) and len(chain) >= 2
    # 3.35B fits 24GB on every leg (Phase-1 DPO ran L4): L4 first.
    assert _const(APP, "TRAIN_GPU")[0] == "L4"
    assert _const(APP, "DUAL_GPU")[0] == "L4"
    assert _const(APP, "INFER_GPU")[0] == "L4"


def test_chat_template_parity_with_phase1():
    phase1 = ROOT / "infra" / "modal" / "finetune_aya73.py"
    assert _const(APP, "CHAT_TEMPLATE") == _const(phase1, "CHAT_TEMPLATE")
    assert _const(APP, "BASE_MODEL") == _const(phase1, "BASE_MODEL")
    assert _const(APP, "BASE_REVISION") == _const(phase1, "BASE_REVISION")
    assert _const(APP, "SEED") == 73


def test_cold_start_stacks():
    stacks = _const(APP, "STACKS")
    # Phase-1 DPO trained IN PLACE on the SFT adapter: it merges
    # straight onto base (never sft+dpo stacked).
    assert stacks["dpo"] == ["dpo"]
    assert stacks["repair"] == ["dpo", "repair"]
    assert stacks["kto"] == ["dpo", "kto"]
    assert stacks["ipo"] == ["sft", "ipo"]
    assert stacks["orpo"] == ["orpo"]  # single-stage from base
    assert stacks["rft"] == ["dpo", "rft"]
    assert stacks["grpo"] == ["dpo", "repair", "grpo"]
    assert set(stacks) == {"base", "sft", "dpo", "repair", "kto",
                           "ipo", "orpo", "rft", "grpo"}
    repos = _const(APP, "HF_REPOS")
    assert set(repos) == {"repair", "kto", "ipo", "orpo", "rft", "grpo"}


def test_inline_reward_mirrors_package():
    from hsea.rlvr.rewards import rule_reward_dispatch

    mirror = _func(APP, "_dispatch_reward")
    cases = [
        ("P1: task the nearest unit now.", "P1"),
        ("P2 noted for the record.", "P2"),
        ("P3: notify before acting, revisit.", "P3"),
        ("done.\nDECISION: comply", "P1"),
        ("", "P1"),
    ]
    for comp, gold in cases:
        assert mirror([comp], [gold])[0] == \
            rule_reward_dispatch(comp, gold)


def test_stage_cmds_wired():
    src = APP.read_text()
    for cmd in ("repair", "kto", "ipo", "orpo", "grpo", "rft",
                "rft-train", "generate", "generate_hf", "stage4",
                "push", "verify"):
        assert f'cmd == "{cmd}"' in src
    for fn in ("def repair(", "def kto(", "def ipo(", "def orpo(",
               "def grpo(", "def rft(", "def rft_train(",
               "def generate(", "def generate_hf(",
               "def push_followup(", "def verify(",
               "def _orpo_loss(", "def _mean_response_logprob(",
               "def _log_odds(", "def _orpo_tokenize(",
               "def _flywheel_dispatch("):
        assert fn in src
    assert 'loss_type="ipo"' in src
    # TRL 1.17 dropped ORPO: custom HF Trainer subclass + tested loss.
    assert "class ORPOTrainer(_ORPOTrainerMixin, Trainer)" in src
    assert "def orpo_compute_loss(" in src
    assert "from trl import ORPOConfig" not in src
    assert "remove_unused_columns=False" in src  # custom ORPO columns
    assert "GRPOTrainer" in src and "KTOTrainer" in src
    assert "rft_train.remote(" in src
    assert "LLM(" in src  # vLLM offline generation


def test_rft_grpo_use_composed_schema():
    src = APP.read_text()
    assert '"flywheel-v2"' in src
    assert 'provenance"]["priority"' in src or \
        "provenance')['priority']" in src
    assert "rec['situation']" not in src
    assert 'rec["situation"]' not in src
    assert "num_return_sequences=n_samples" not in src
    assert "num_return_sequences=1" in src  # sequential RFT sampling
    # Removed from GRPOConfig: never passed as a kwarg; verify() guards it.
    assert "max_prompt_length=" not in src
    assert '"max_prompt_length" not in inspect.signature(' in src


def test_entrypoint_per_command_defaults():
    src = APP.read_text()
    assert "epochs=epochs if epochs > 0 else 1" in src  # repair/kto
    assert "max_steps if max_steps > 0 else 100" in src  # grpo
