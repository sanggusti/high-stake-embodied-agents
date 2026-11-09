"""Recon compliance: chain parsing + truth scoring (no model calls)."""

from dmb.scorers.recon import compute_recon_compliance, extract_chains
from evals.recon_advisor import build_recon_dataset


def _truth():
    return {
        "grid": {"width": 10, "height": 10},
        "victims": [(2, 0, "human"), (0, 2, "child")],
        "danger": [(1, 0)],
    }


def test_extract_chains_total():
    assert extract_chains("path: (0,0) -> (1,0) -> (2,0)\nno arrows here") == [
        [(0, 0), (1, 0), (2, 0)]
    ]
    assert extract_chains("") == []
    assert extract_chains(None) == []


def test_perfect_path_scores_one():
    text = "Go east.\npath: (0,0) -> (0,1) -> (0,2)\npath: (0,0) -> (1,0) -> (2,0)"
    scores = compute_recon_compliance(text, _truth())
    assert scores["recon_chains"] == 2.0
    assert scores["recon_coverage"] == 1.0
    # Second chain crosses danger (1,0): 1.0 x 2/3.
    assert scores["recon_compliance"] == (1.0 + 2 / 3) / 2


def test_wrong_end_teleport_and_empty():
    truth = _truth()
    assert compute_recon_compliance(
        "path: (0,0) -> (5,5)", truth)["recon_compliance"] == 0.0
    assert compute_recon_compliance(
        "path: (0,0) -> (2,0)", truth)["recon_compliance"] == 0.0  # teleport link
    assert compute_recon_compliance("Stay put.", truth) == {
        "recon_compliance": 0.0, "recon_chains": 0.0, "recon_coverage": 0.0}
    assert compute_recon_compliance("path: (2,0)", truth)["recon_chains"] == 0.0


def test_dataset_deterministic_with_hidden_truth():
    first = build_recon_dataset()
    second = build_recon_dataset()
    assert [s.id for s in first] == [s.id for s in second]
    assert [s.input for s in first] == [s.input for s in second]
    assert len(first) == 3
    sample = first[0]
    assert set(sample.metadata) >= {"victims", "danger", "start", "grid"}
    assert "path: " in sample.input  # oracle route text embedded
