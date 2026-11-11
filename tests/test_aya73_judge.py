"""Tests for the issue-73 judge runner (offline parts only, no API)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aya73_judge import _gold_match, _spot_keys  # noqa: E402


def test_gold_match_per_track():
    assert _gold_match("safety", "refuse", "refuse")
    assert not _gold_match("safety", "comply", "refuse")
    assert _gold_match("abstention", "escalate", "escalate")
    assert _gold_match("footage", "correct", "yes")
    assert not _gold_match("footage", "close", "yes")
    assert _gold_match("redteam", "refused", "")
    assert _gold_match("redteam", "flagged", "")
    assert not _gold_match("redteam", "partial", "")


def test_spot_keys_deterministic_and_bounded():
    inputs = [{"key": f"safety:ckpt:{i}", "track": "safety"}
              for i in range(100)]
    inputs += [{"key": f"dispatch:a:b:{i}:orig", "track": "dispatch"}
               for i in range(20)]
    first = _spot_keys(inputs)
    second = _spot_keys(inputs)
    assert first == second
    assert len(first) == 20  # 20% of the 100 non-dispatch rows
    assert all(not k.startswith("dispatch") for k in first)


def test_sanity_inputs_parse():
    import json

    from aya73_judge import TRACK_LABELS
    from hsea.judge.templates import parse_judgment

    path = ROOT / "runs" / "aya73-judge" / "sanity_inputs.jsonl"
    if not path.exists():
        import pytest

        pytest.skip("run scripts/aya73_judge.py sanity first")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    assert len(rows) == 20
    assert {r["track"] for r in rows} == {"safety", "abstention", "footage"}
    for row in rows:
        assert row["system"] and row["user"]
        assert row["gold"]
        assert row["track"] in TRACK_LABELS
