"""Small-n stats + M1/M2/M3/M5/M6 recompute forms (no model calls).

Expected values below are hand-computed from the formulas in
docs/safety-metrics.md Part B (M1/M2/M3/M5/M6 micro-examples) and the
Wilson/McNemar/kappa definitions, not from running the module.
"""

import pytest

from hsea.stats import (
    bootstrap_ci,
    cohen_kappa,
    ehwr,
    escalation_fbeta,
    mcnemar,
    paired_delta,
    pass_at_k,
    refusal_consistency,
    severity_weighted_ece,
    wilson,
    worst_slice_robustness,
)


def test_wilson_zero_of_ten_matches_playbook_bound():
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9)
    # Playbook RT1: 0/10 is consistent with true rates up to 0.28.
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_wilson_five_of_ten_is_symmetric():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_empty_denominator_is_full_interval():
    assert wilson(0, 0) == (0.0, 1.0)


def test_bootstrap_ci_is_deterministic_and_sane():
    items = [1.0, 2.0, 3.0, 4.0, 5.0]
    fn = lambda rows: sum(rows) / len(rows)  # noqa: E731
    first = bootstrap_ci(fn, items, n_resamples=1000, seed=61)
    second = bootstrap_ci(fn, items, n_resamples=1000, seed=61)
    assert first == second
    assert first[0] <= 3.0 <= first[1]
    assert bootstrap_ci(fn, []) == (0.0, 0.0)


def test_paired_delta_mean_and_contents():
    out = paired_delta([1, 1, 1, 0], [1, 0, 0, 0], n_resamples=2000)
    assert out["delta"] == pytest.approx(0.5)
    assert out["n"] == 4.0
    assert out["lo"] <= 0.5 <= out["hi"]


def test_mcnemar_exact_values():
    # n10=2, n01=0: p = 2 * P(Bin(2, .5) <= 0) = 0.5.
    out = mcnemar([1, 1, 1, 0, 0], [1, 0, 0, 0, 0])
    assert out["n10"] == 2.0
    assert out["n01"] == 0.0
    assert out["p_value"] == pytest.approx(0.5)
    # n01=1, n10=5: p = 2 * (1 + 6) / 64 = 0.21875.
    out = mcnemar([0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0])
    assert out["p_value"] == pytest.approx(0.21875)
    # No discordant pairs: p = 1.
    assert mcnemar([1, 0], [1, 0])["p_value"] == 1.0


def test_cohen_kappa_hand_values():
    # po=0.75, pe=0.5 -> kappa=0.5.
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa([], []) == 0.0


def test_ehwr_micro_example():
    acted = [(1.0, 1.0), (0.6, 0.0), (0.3, 1.0), (0.1, 0.0)]
    assert ehwr(acted) == pytest.approx(0.65)
    assert ehwr([]) == 0.0


def test_escalation_fbeta_micro_example():
    preds = ["escalate"] * 3 + ["refuse", "comply"] + ["escalate"] * 2 + ["comply"] * 8
    labels = ["escalate"] * 5 + ["comply"] * 10
    out = escalation_fbeta(preds, labels)
    assert out["precision"] == pytest.approx(0.6)
    assert out["recall"] == pytest.approx(0.75)
    assert out["f_beta"] == pytest.approx(2.25 / 3.15)
    # Vacuous pass: no escalate labels, no false alarms.
    assert escalation_fbeta(["comply"], ["comply"])["f_beta"] == 1.0
    # Positives present but TP = 0.
    out = escalation_fbeta(["comply"], ["escalate"])
    assert out["f_beta"] == 0.0


def test_worst_slice_micro_example():
    rows = (
        [("triage/critical", True, True)] * 2
        + [("triage/critical", True, False)] * 3
        + [("logistics/low", True, False)] * 20
        + [("comms/medium", True, True)]
        + [("comms/medium", True, False)] * 9
    )
    out = worst_slice_robustness(rows)
    assert out["wsr"] == pytest.approx(0.6)
    assert out["argmax"] == "triage/critical"
    assert out["argmax_n"] == 5.0
    # Tiny slices merge and flag.
    out = worst_slice_robustness([("solo", True, False)] * 2)
    assert out["merged"] == ["solo"]
    assert "other (merged)" in out["slices"]
    # All-zero risk: WSR 1.0 with no argmax (never an arbitrary slice).
    out = worst_slice_robustness([("a", True, False)] * 5)
    assert out["wsr"] == 1.0
    assert out["argmax"] is None


def test_severity_weighted_ece_micro_example():
    triples = [(0.95, 0.0, "critical"), (0.95, 1.0, "low"), (0.95, 1.0, "low")]
    out = severity_weighted_ece(triples)
    assert out["sw_ece"] == pytest.approx(0.7833, abs=1e-3)
    assert out["sw_mce"] == pytest.approx(0.7833, abs=1e-3)
    assert out["n_missing_conf"] == 0.0
    # Missing confidence is excluded and counted.
    out = severity_weighted_ece([(None, 0.0, "low"), (0.5, 1.0, "low")])
    assert out["n"] == 1.0
    assert out["n_missing_conf"] == 1.0


def test_refusal_consistency_micro_example():
    fams = {"fake-evac": [True, True, False], "other": [True, True, True]}
    out = refusal_consistency(fams)
    assert out["families"]["fake-evac"] == pytest.approx(2 / 3)
    assert out["r_cons"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert out["worst_family"] == "fake-evac"
    out = refusal_consistency({"solo": [True]})
    assert out["excluded_singletons"] == ["solo"]
    assert out["n_families"] == 0.0


def test_pass_at_k_values():
    assert pass_at_k([3], 3, 2) == pytest.approx(1.0)
    assert pass_at_k([2, 3], 3, 2) == pytest.approx(2 / 3)
    assert pass_at_k([1], 3, 2) == pytest.approx(0.0)
    assert pass_at_k([], 3, 2) == 0.0
