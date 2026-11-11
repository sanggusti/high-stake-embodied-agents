"""Tests for the issue-73 Tiny Aya mix composer (seed 73)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compose_aya73 import (  # noqa: E402
    abstention_gold,
    compose,
    safety_demo_response,
    split_safety,
)
from hsea.safety.metrics import parse_decision  # noqa: E402


def _load(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_safety_split_gate_never_trains():
    from hsea.safety.data import ITEMS

    from compose_aya73 import GATE_BENCHES

    train, evl = split_safety()
    assert len(train) + len(evl) == len(ITEMS)
    assert {t[3] for t in train}.isdisjoint({e[3] for e in evl})
    assert all(t[0] not in GATE_BENCHES for t in train)
    gate = [e for e in evl if e[0] in GATE_BENCHES]
    assert len(gate) == 26  # full HarmBench (14) + AIR-Bench (12) lists


def test_safety_demos_carry_their_label():
    for label in ("refuse", "escalate", "abstain", "comply"):
        text = safety_demo_response(label, "Do the thing now.")
        assert parse_decision(text) == label


def test_abstention_gold_has_action_footer():
    rec = {"label": "escalate", "rationale": "Stakes are high."}
    assert abstention_gold(rec).endswith("ACTION: escalate\nCONFIDENCE: 85")


def test_compose_deterministic_and_disjoint(tmp_path):
    first = compose(tmp_path / "a")
    second = compose(tmp_path / "b")
    assert first["counts"] == second["counts"]
    for name in ("sft_train.jsonl", "dpo_train.jsonl",
                 "eval_safety.jsonl", "eval_abstention.jsonl",
                 "eval_redteam.jsonl", "eval_footage.jsonl",
                 "eval_dispatch.jsonl"):
        a = (tmp_path / "a" / name).read_bytes()
        b = (tmp_path / "b" / name).read_bytes()
        assert a == b, name

    # Footage holdout ids are disjoint from SFT provenance by construction:
    # the holdout carves floodnet rows out of the train file.
    sft = _load(tmp_path / "a" / "sft_train.jsonl")
    assert first["counts"]["sft_train"] == len(sft)
    assert first["counts"]["sft_flywheel"] == 8613
    assert first["counts"]["sft_safety"] == 26
    assert first["counts"]["sft_abstention"] == 28
    assert first["counts"]["dpo_train"] == 7054
    assert first["counts"]["eval_safety"] == 78
    assert first["counts"]["eval_abstention"] == 52
    assert first["counts"]["eval_redteam"] == 30
    assert first["counts"]["eval_footage"] == 60
    assert first["counts"]["eval_dispatch"] == 24

    # Eval safety carries gate/trained flags; gate rows never trained.
    safety = _load(tmp_path / "a" / "eval_safety.jsonl")
    assert sum(1 for r in safety if r["gate"]) == 26
    assert all(not r["trained"] for r in safety if r["gate"])

    # Every SFT row is chat messages ending in an assistant turn.
    for row in sft[:50]:
        roles = [m["role"] for m in row["messages"]]
        assert roles[-1] == "assistant"
        assert "user" in roles

    # Every DPO row has a non-empty chosen/rejected pair.
    dpo = _load(tmp_path / "a" / "dpo_train.jsonl")
    for row in dpo[:50]:
        assert row["chosen"] and row["rejected"]
        assert row["chosen"] != row["rejected"]
