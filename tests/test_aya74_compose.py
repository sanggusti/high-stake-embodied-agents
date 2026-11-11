"""Composer tests (#74): mix counts, contract parse, clearance."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from scripts.compose_aya74 import (  # noqa: E402
    BLOCKED_SOURCES,
    abstention_demo,
    check_clearance,
    compose_dpo,
    compose_sft,
    safety_demo,
    safety_violation,
)

from hsea.safety.metrics import classify_response  # noqa: E402
from hsea.scorers.abstention import parse_verdict  # noqa: E402
from hsea.safety.data import ITEMS  # noqa: E402


def test_compose_counts():
    sft_rows, sft_stats = compose_sft()
    dpo_rows, dpo_stats = compose_dpo()
    assert sft_stats == {"flywheel": 8673, "safety": 78, "abstention": 80,
                         "total": 8831}
    assert dpo_stats == {"flywheel": 7000, "safety": 78, "total": 7078}
    assert len(sft_rows) == 8831
    assert len(dpo_rows) == 7078


def test_sft_rows_wellformed():
    sft_rows, _ = compose_sft()
    for row in sft_rows:
        roles = [m["role"] for m in row["messages"]]
        assert set(roles) <= {"user", "assistant"}, roles
        assert roles[0] == "user" and roles[-1] == "assistant"
        assert all(isinstance(m["content"], str) and m["content"].strip()
                   for m in row["messages"])
        assert "tool_calls" not in json.dumps(row["messages"])
        assert row["source"]
        assert row["provenance"]["mix"] == "aya74-sft"


def test_l2_flatten_preserves_tool_calls():
    sft_rows, _ = compose_sft()
    l2_rows = [r for r in sft_rows if r["source"] == "l2"]
    assert len(l2_rows) == 251
    joined = json.dumps([r["messages"] for r in l2_rows])
    assert "[tool-call]" in joined and "[tool result]" in joined


def test_safety_demos_parse_to_gold():
    for _, category, label, prompt in ITEMS:
        demo = safety_demo(label, prompt, category)
        assert classify_response(demo) == label, (label, demo)


def test_safety_violations_miss_gold():
    for _, category, label, prompt in ITEMS:
        bad = safety_violation(label, prompt, category)
        assert classify_response(bad) != label, (label, bad)


def test_abstention_demos_parse_to_gold():
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for line in open(ROOT / "data" / "abstention" / name):
            rec = json.loads(line)
            action, conf = parse_verdict(
                abstention_demo(rec["label"], rec.get("rationale", "")))
            assert action == rec["label"], rec["id"]
            assert conf is not None


def test_dpo_rows_wellformed():
    dpo_rows, _ = compose_dpo()
    for row in dpo_rows:
        assert row["prompt"].strip()
        assert row["chosen"].strip() and row["rejected"].strip()
        assert row["chosen"] != row["rejected"]
        assert row["provenance"]["mix"] == "aya74-dpo"


def test_clearance_no_blocked_sources():
    clearance = check_clearance()
    for src in clearance["flywheel_sources"]:
        norm = src.lower().replace("_", "").replace("-", "")
        blocked = {b.replace("-", "") for b in BLOCKED_SOURCES}
        assert norm not in blocked, src


def test_compose_cli_writes_manifest(tmp_path):
    out = tmp_path / "aya74"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compose_aya74.py"),
         "--out", str(out)], capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sft"]["total"] == 8831
    assert manifest["dpo"]["total"] == 7078
    assert "overlap_disclosure" in manifest
    assert sum(1 for _ in open(out / "sft.jsonl")) == 8831
    assert sum(1 for _ in open(out / "dpo.jsonl")) == 7078
