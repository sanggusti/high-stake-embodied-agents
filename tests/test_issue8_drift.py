"""Tests for issue-8 drift audit helpers (pure dicts, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import issue8_drift as drift
from hsea.safety.data import ITEMS


def _row(completion, bench="benign", label="comply", input="x",
         severity="high", value=0.0, rid="r"):
    return {"id": rid, "input": input, "completion": completion,
            "metadata": {"bench": bench, "label": label,
                         "severity": severity},
            "value": value, "score_metadata": {}}


def test_over_refusal_rate_counts_only_comply_rows():
    rows = [_row("DECISION: refuse\nCONFIDENCE: 95\nR: no.",
                 label="comply"),
            _row("DECISION: comply\nCONFIDENCE: 90\nR: yes.",
                 label="comply"),
            _row("DECISION: refuse\nCONFIDENCE: 95\nR: no.",
                 label="refuse")]
    out = drift.over_refusal_rate(rows)
    assert out == {"n": 2, "rate": 0.5}


def test_agentharm_harm_score_pairs_misuse_and_benign():
    rows = [_row("DECISION: comply\nCONFIDENCE: 90\nR: filed.",
                 bench="agentharm", label="refuse"),
            _row("DECISION: refuse\nCONFIDENCE: 95\nR: no.",
                 bench="agentharm", label="refuse"),
            _row("DECISION: comply\nCONFIDENCE: 90\nR: drafted.",
                 bench="agentharm", label="comply")]
    out = drift.agentharm_harm_score(rows)
    assert out["misuse_n"] == 2
    assert out["harm_score"] == 0.5
    assert out["benign_compliance"] == 1.0


def test_safety_vectors_map_inputs_to_heldout():
    bench, _cat, label, prompt = ITEMS[0]
    rows = [_row("DECISION: refuse\nCONFIDENCE: 95\nR: no.",
                 bench=bench, label=label, input=prompt)]
    preds, labels, confs, benches = drift._safety_vectors(rows)
    assert (preds, labels, benches) == (["refuse"], [label], [bench])
    assert confs == [0.95]
    gid = f"{bench}:0"
    kept = drift._safety_vectors(rows, {gid})
    assert len(kept[0]) == 1
    dropped = drift._safety_vectors(rows, {"other:9"})
    assert dropped[0] == []


def test_severity_asr_groups_by_severity():
    rows = [_row("x", severity="high", value=1.0),
            _row("x", severity="high", value=0.0),
            _row("x", severity="low", value=0.0)]
    out = drift._severity_asr(rows)
    assert out == {"high": 0.5, "medium": None, "low": 0.0}


def test_benign_flat_check_tolerance():
    summary = {"runs": {
        "base/flywheel_test": {"metrics": {"mean_composite": 0.60}},
        "dpo/flywheel_test": {"metrics": {"mean_composite": 0.62}},
        "base/floodnet_vqa": {"metrics": {"mean_composite": 0.70}},
        "dpo/floodnet_vqa": {"metrics": {"mean_composite": 0.70}},
        "base/ladi_v2": {"metrics": {"mean_composite": 0.50}},
        "dpo/ladi_v2": {"metrics": {"mean_composite": 0.40}},
    }}
    out = drift.benign_flat_check(summary, "dpo")
    assert out["flywheel_test"]["flat"] is True
    assert out["ladi_v2"]["flat"] is False
    assert out["flat_all"] is False
