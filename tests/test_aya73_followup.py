"""Follow-up prompt/score/drift pins (#73 stages 3-6)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aya73_drift  # noqa: E402
import aya73_followup as fu  # noqa: E402

RUN_DIR = ROOT / "runs" / "aya73-followup"


def _prompts() -> list[dict]:
    with open(RUN_DIR / "eval_prompts.jsonl") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_prompts_layout():
    prompts = _prompts()
    assert len(prompts) == 322
    by_track: dict[str, int] = {}
    for p in prompts:
        by_track[p["track"]] = by_track.get(p["track"], 0) + 1
    assert by_track == {"safety": 78, "abstention": 80, "redteam": 30,
                       "xstest": 50, "dispatch": 24, "footage": 60}
    keys = [p["key"] for p in prompts]
    assert len(set(keys)) == len(keys)
    assert all(p["max_tokens"] > 0 for p in prompts)


def test_dispatch_golds_join():
    from hsea.rlvr.rewards import priority_of

    prompts = [p for p in _prompts() if p["track"] == "dispatch"]
    assert len(prompts) == 24
    assert all(p["gold"] in ("P1", "P2", "P3") for p in prompts)
    assert all(p["meta"]["priority"] == p["gold"] for p in prompts)
    assert priority_of is not None


def test_xstest_uses_safety_system():
    from hsea.safety.data import SAFETY_SYSTEM

    prompts = [p for p in _prompts() if p["track"] == "xstest"]
    assert prompts and all(p["system"] == SAFETY_SYSTEM for p in prompts)
    assert sum(1 for p in prompts if p["gold"] == "comply") == 40
    assert sum(1 for p in prompts if p["gold"] == "refuse") == 10


def _fixture(tmp_path: Path) -> None:
    prompts = [
        {"key": "safety:harmbench-1", "track": "safety", "system": "s",
         "prompt": "harm", "gold": "refuse", "max_tokens": 8,
         "meta": {"bench": "harmbench", "category": "c",
                  "gate": True, "trained": False}},
        {"key": "safety:benign-1", "track": "safety", "system": "s",
         "prompt": "benign", "gold": "comply", "max_tokens": 8,
         "meta": {"bench": "benign", "category": "c",
                  "gate": False, "trained": True}},
        {"key": "redteam:rt-1", "track": "redteam", "system": "s",
         "prompt": "attack", "gold": "refused", "max_tokens": 8,
         "meta": {"family": "jailbreak", "variant": "v",
                  "severity": "high", "markers": []}},
        {"key": "dispatch:t-01", "track": "dispatch", "system": "s",
         "prompt": "sit", "gold": "P1", "max_tokens": 8,
         "meta": {"priority": "P1"}},
    ]
    with open(tmp_path / "eval_prompts.jsonl", "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    base = {"safety:harmbench-1": "DECISION: refuse\nCONFIDENCE: 95",
            "safety:benign-1": "DECISION: comply\nCONFIDENCE: 90",
            "redteam:rt-1": "I can't help with that.",
            "dispatch:t-01": "P1: task the nearest unit now."}
    grpo = dict(base)
    rft = dict(base)
    rft["dispatch:t-01"] = "Log this and revisit in the routine cycle."
    for ckpt, comps in (("base", base), ("grpo", grpo), ("rft", rft)):
        with open(tmp_path / f"eval_{ckpt}.jsonl", "w") as fh:
            for key, completion in comps.items():
                fh.write(json.dumps({"key": key,
                                     "completion": completion}) + "\n")


def test_score_and_rewards_gates(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setattr(fu, "RUN_DIR", tmp_path)
    monkeypatch.setattr(fu, "CHECKPOINTS", ("base", "grpo", "rft"))
    fu.cmd_score()
    det = json.loads((tmp_path / "det_scores.json").read_text())
    assert det["checkpoints"]["base"]["safety"]["rate"] == 1.0
    assert det["checkpoints"]["base"]["gate"]["rate"] == 1.0
    fu.cmd_rewards()
    rew = json.loads((tmp_path / "dispatch_rewards.json").read_text())
    assert rew["checkpoints"]["grpo"]["mean_reward"] == 1.2
    assert rew["checkpoints"]["rft"]["mean_reward"] == 0.0  # hedge floors
    assert rew["grpo_beats_rft"] is True


def test_harm_score_shared_with_74():
    import aya74_drift

    assert aya73_drift.harm_score is aya74_drift.harm_score
    assert aya73_drift.harm_score("DECISION: refuse") == 0.0


def test_verdict_drift_panel(tmp_path):
    prompts = [
        {"key": "abstention:a1", "track": "abstention", "gold": "comply",
         "meta": {}},
        {"key": "abstention:a2", "track": "abstention",
         "gold": "escalate", "meta": {}},
    ]
    comps = {"dpo": {
        "abstention:a1": "x\nACTION: comply\nCONFIDENCE: 70\ny\n"
                         "ACTION: escalate\nCONFIDENCE: 55",
        "abstention:a2": "z\nACTION: escalate\nCONFIDENCE: 80",
    }}
    panel = aya73_drift.abstention_drift_panel(prompts, comps)
    cell = panel["checkpoints"]["dpo"]
    assert cell["multi_rate"] == 0.5
    assert cell["first_acc"] == 1.0
    assert cell["last_acc"] == 0.5
