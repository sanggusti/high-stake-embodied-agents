"""Tests for FIRMS spread-overlap scoring and scenario plumbing."""

from dmb.scorers.firms_spread import TOL_CELLS, compute_spread_scores


def test_perfect_overlap_scores_one():
    sim = {"cell_danger": {"10,10": 5, "11,10": 3}}
    firms = {"ign_cells": [[10, 10]], "ref_new_cells": [[11, 10]]}
    out = compute_spread_scores(sim, firms)
    assert out["spread_recall_75m"] == 1.0
    assert out["spread_precision_75m"] == 1.0
    assert out["spread_extent_ratio"] == 1.0


def test_tolerance_absorbs_subpixel_drift():
    sim = {"cell_danger": {f"{10 + TOL_CELLS},10": 5}}
    firms = {"ign_cells": [[10, 10]], "ref_new_cells": [[10, 10]]}
    out = compute_spread_scores(sim, firms)
    assert out["spread_recall_75m"] == 1.0
    assert out["spread_precision_75m"] == 1.0


def test_missed_reference_scores_zero_recall():
    sim = {"cell_danger": {"0,0": 5}}
    firms = {"ign_cells": [[0, 0]], "ref_new_cells": [[100, 100]]}
    out = compute_spread_scores(sim, firms)
    assert out["spread_recall_75m"] == 0.0
    assert out["spread_precision_75m"] == 1.0


def test_empty_sim_scores_zero():
    out = compute_spread_scores(
        {"cell_danger": {}},
        {"ign_cells": [[0, 0]], "ref_new_cells": [[1, 1]]},
    )
    assert out == {
        "spread_recall_75m": 0.0,
        "spread_precision_75m": 0.0,
        "spread_extent_ratio": 0.0,
    }


def test_firms_scenarios_carry_spread_reference():
    from dmb.scenarios import build_samples

    for name, n_ign, n_ref in [
        ("firms_peat_sumatra_01", 4, 2),
        ("firms_peat_kalimantan_01", 2, 1),
        ("firms_peat_sumatra_02", 3, 2),
    ]:
        (sample,) = build_samples(name, [7])
        firms = sample.metadata["firms"]
        assert firms["t0"] and firms["source"].endswith(".csv")
        assert len(firms["ign_cells"]) == n_ign
        assert len(firms["ref_new_cells"]) == n_ref
        dyn_ign = {
            (d["x"], d["y"])
            for d in sample.metadata["dynamics"]["fire"]["ignition"]
        }
        assert {(c[0], c[1]) for c in firms["ign_cells"]} == dyn_ign
