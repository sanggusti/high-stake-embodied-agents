"""Stage 4/5 tests (#74): repair mixes, KTO labels, RLVR rewards."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compose_aya74_repair import (  # noqa: E402
    debias,
    stratified,
    unified_demo,
)
from hsea.rlvr.rewards import (  # noqa: E402
    field_match_reward,
    has_footer,
    is_hedged,
    mask_tool_spans,
    passes_verifier,
    priority_of,
    rule_reward_dispatch,
)


def test_unified_demo_footer():
    for label in ("escalate", "abstain", "comply"):
        text = unified_demo(label, "Rationale here.")
        assert f"DECISION: {label}" in text
        assert "ACTION:" not in text
        assert "CONFIDENCE:" in text


def test_debias_strips_prefix_both_styles():
    assert debias("PRIORITY 1 \u2014 IMMEDIATE DISPATCH. Task unit 7 now.") == \
        "Task unit 7 now."
    assert debias("PRIORITY 3 \u2014 MONITOR. Log and revisit.") == \
        "Log and revisit."
    assert debias("No prefix here.") == "No prefix here."


def test_stratified_seeded_and_proportional():
    rows = [{"priority": "P1", "i": i} for i in range(10)]
    rows += [{"priority": "P2", "i": i} for i in range(30)]
    rows += [{"priority": "P3", "i": i} for i in range(20)]
    a = stratified(rows, 12)
    b = stratified(rows, 12)
    assert [r["i"] for r in a] == [r["i"] for r in b]
    counts = {}
    for r in a:
        counts[r["priority"]] = counts.get(r["priority"], 0) + 1
    assert sum(counts.values()) == 12
    assert counts["P2"] >= counts["P1"]  # proportional-ish


def _repair_sft():
    return [json.loads(line) for line in
            open(ROOT / "data" / "aya74" / "repair_sft.jsonl")]


def test_repair_mix_counts():
    sft = _repair_sft()
    assert len(sft) == 438
    sources = {}
    for r in sft:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    assert sources == {"hsea-safety": 78, "hsea-abstention": 80,
                       "flywheel-dispatch-anchor": 200, "wj-method": 80}
    dpo = [json.loads(line) for line in
           open(ROOT / "data" / "aya74" / "repair_dpo.jsonl")]
    assert len(dpo) == 578
    kto = [json.loads(line) for line in
           open(ROOT / "data" / "aya74" / "kto.jsonl")]
    assert len(kto) == 160
    assert sum(1 for r in kto if r["label"]) == 80
    assert sum(1 for r in kto if not r["label"]) == 80


def test_repair_footer_contract():
    import re

    vals = set()
    for rec in _repair_sft():
        for msg in rec["messages"]:
            if msg["role"] != "assistant":
                continue
            assert "ACTION:" not in msg["content"]
            for v in re.findall(r"DECISION:\s*(\S+)", msg["content"]):
                vals.add(v.strip().lower())
    assert vals <= {"refuse", "escalate", "abstain", "comply"}
    assert len(vals) == 4
    # Dispatch anchors carry no footer at all.
    for rec in _repair_sft():
        if rec["source"] != "flywheel-dispatch-anchor":
            continue
        text = rec["messages"][-1]["content"]
        assert "DECISION:" not in text and "ACTION:" not in text


def test_repair_dpo_debiased():
    for line in open(ROOT / "data" / "aya74" / "repair_dpo.jsonl"):
        rec = json.loads(line)
        if rec["source"] != "flywheel-dispatch-debiased":
            continue
        assert not rec["chosen"].startswith("PRIORITY")
        assert not rec["rejected"].startswith("PRIORITY")
        assert rec["chosen"] and rec["rejected"]


def test_repair_disjoint_from_held_out():
    xs = {json.loads(line)["prompt"] for line in
          open(ROOT / "data" / "xstest" / "lookalikes.jsonl")}
    train = set()
    for rec in _repair_sft():
        for msg in rec["messages"]:
            if msg["role"] == "user":
                train.add(msg["content"])
    assert not (xs & train)
    held = {json.loads(line)["situation"] for line in
            open(ROOT / "data" / "dispatch-pairs" / "dpo.jsonl")}
    assert not any(any(h in m["content"] for h in held)
                   for rec in _repair_sft() for m in rec["messages"])


def test_priority_and_fields():
    assert priority_of("PRIORITY 1 \u2014 go now") == "P1"
    assert priority_of("set P2 for this") == "P2"
    assert priority_of("no priority here") is None
    good = "P1: task the nearest unit now, confirm en route."
    assert field_match_reward(good, "P1") == 1.0
    assert field_match_reward("P1 noted for the record.", "P1") == 0.5
    assert field_match_reward(good, "P2") == 0.0
    assert has_footer("done.\nDECISION: comply") is True
    assert has_footer(good) is False
    assert is_hedged("Log this and revisit in the routine cycle.") is True
    assert is_hedged(good) is False


def test_rule_reward_and_verifier():
    good = "P1: task the nearest unit now, confirm en route."
    assert rule_reward_dispatch(good, "P1") == 1.2
    assert passes_verifier(good, "P1") is True
    assert passes_verifier(good + "\nDECISION: comply", "P1") is False
    assert passes_verifier("P2: task unit 9.", "P1") is False
    hedged = "P1: notify before acting, then revisit."
    assert rule_reward_dispatch(hedged, "P1") < 1.0


def test_mask_tool_spans():
    mask = mask_tool_spans([("prompt", 3), ("assistant", 2), ("tool", 4),
                            ("assistant", 1)])
    assert mask == [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    try:
        mask_tool_spans([("weird", 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
