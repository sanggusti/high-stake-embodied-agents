"""Tests for issue-8 eval summarizers (pure dicts, no GPU)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue8_evals as ev


def _row(value=1.0, metadata=None, score_metadata=None):
    return {"id": "x", "input": "i", "completion": "c", "target": "t",
            "metadata": metadata or {},
            "value": value, "answer": "",
            "score_metadata": score_metadata or {}}


def test_summarize_footage_kinds():
    rows = [_row(1.0, {"kind": "vqa"}), _row(0.0, {"kind": "vqa"}),
            _row(1.0, {"kind": "counting"})]
    out = ev.summarize_footage(rows)
    assert out["n"] == 3
    assert out["mean_composite"] == 2 / 3
    assert out["by_kind"]["vqa"] == {"n": 2, "mean": 0.5}
    assert out["by_kind"]["counting"]["mean"] == 1.0


def test_summarize_safety_benches_and_refusal():
    rows = [
        _row(1.0, {"bench": "harmbench"}, {"refused": 1.0}),
        _row(0.0, {"bench": "harmbench"}, {"refused": 0.0}),
        _row(1.0, {"bench": "benign"}, {"refused": 0.0}),
    ]
    out = ev.summarize_safety(rows)
    assert out["match_rate"] == 2 / 3
    assert out["by_bench"]["harmbench"]["match"] == 0.5
    assert out["refusal"]["harmbench"]["rate"] == 0.5
    assert out["refusal"]["airbench"] == {"n": 0, "rate": 0.0}


def test_summarize_redteam_asr_and_review():
    rows = [
        _row(0.0, {"family": "jailbreak"}, {"rt_needs_review": 0.0}),
        _row(1.0, {"family": "jailbreak"}, {"rt_needs_review": 1.0}),
        _row(0.0, {"family": "misuse"}, {"rt_needs_review": 0.0}),
    ]
    out = ev.summarize_redteam(rows)
    assert out["asr"] == 1 / 3
    assert out["needs_review"] == 1
    assert out["by_family"]["jailbreak"]["asr"] == 0.5


def test_summarize_mission_means():
    rows = [_row(0.8, {}, {"composite": 0.8, "rescue_recall": 1.0}),
            _row(0.4, {}, {"composite": 0.4, "rescue_recall": 0.5})]
    out = ev.summarize_mission(rows)
    assert out["n"] == 2
    assert out["mean_composite"] == pytest.approx(0.6)
    assert out["mean_rescue_recall"] == pytest.approx(0.75)


def test_matrix_covers_required_tracks():
    aliases = [alias for _, alias, _, _ in ev.MATRIX]
    assert aliases == ["flywheel_test", "floodnet_vqa", "ladi_v2",
                       "safety", "redteam"]
    assert set(ev._SUMMARIZERS) == set(aliases)
