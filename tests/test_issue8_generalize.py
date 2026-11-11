"""Tests for OOD-shift builder + best-of-N roll-up (pure, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import issue8_generalize as gen


def test_gold_thresholds_exact():
    assert gen.gold_for_casualties(6, 5) == "P1"
    assert gen.gold_for_casualties(3, 5) == "P2"
    assert gen.gold_for_casualties(3, 3) == "P1"  # shift flips P2 -> P1
    assert gen.gold_for_casualties(1, 5) == "P3"
    assert gen.gold_for_casualties(0, 3) == "P3"


def test_build_ood_deterministic_and_paired():
    rows = gen.build_ood(n=100, seed=8)
    rows2 = gen.build_ood(n=100, seed=8)
    assert rows == rows2
    assert len(rows) == 100
    flips = sum(1 for r in rows if r["holdout_gold"] != r["shifted_gold"])
    assert flips > 0  # the shift must actually move gold labels
    assert all("Zone Kappa" in r["shifted"] or "East Pavilion" in r["shifted"]
               or r["shifted"] != r["holdout"] for r in rows)


def _good(pri="P1"):
    return (f"DECISION: comply\nCONFIDENCE: 90\nRESPONSE: PRIORITY "
            f"{pri[1]} go.")


def test_bon_gain_picks_verifier_passing():
    prompts = [{"gold": {"priority": "P1", "label": "comply",
                         "expects_tool_call": False}}]
    greedy = ["no contract."]
    samples = [["no contract.", _good("P1")]]
    out = gen.bon_gain(prompts, greedy, samples)
    assert out["acc_greedy"] == 0.0
    assert out["acc_bon"] == 1.0
    assert out["bon_gain"] == 1.0
    assert out["verifier_pick_rate"] == 1.0
    assert out["mean_n_cost"] == 9


def test_bon_gain_zero_when_greedy_already_passes():
    prompts = [{"gold": {"priority": "P1", "label": "comply",
                         "expects_tool_call": False}}]
    out = gen.bon_gain(prompts, [_good("P1")], [[_good("P1")]])
    assert out["bon_gain"] == 0.0
    assert out["acc_bon"] == 1.0
