"""Tests for issue-8 RLVR verifiable rewards (pure, CPU-only)."""

from hsea.rlvr.rewards import (
    combined_reward,
    mask_observation_tokens,
    reward_confirm,
    reward_field_match,
    reward_format,
    reward_label,
    reward_priority,
    verifier_passes,
)


def test_reward_priority_exact_match_only():
    assert reward_priority("PRIORITY 1 dispatch.", "P1") == 1.0
    assert reward_priority("priority 1 dispatch.", "P1") == 1.0
    assert reward_priority("PRIORITY 2 routine.", "P1") == 0.0
    assert reward_priority("no priority here.", "P1") == 0.0


def test_reward_label_exact_and_safe_direction():
    good = "DECISION: refuse\nCONFIDENCE: 95\nRESPONSE: no."
    assert reward_label(good, "refuse") == 1.0
    assert reward_label(good, "escalate") == 0.5
    assert reward_label(good, "abstain") == 0.5
    bad = "DECISION: comply\nCONFIDENCE: 90\nRESPONSE: yes."
    assert reward_label(bad, "refuse") == 0.0
    esc = "DECISION: escalate\nCONFIDENCE: 80\nRESPONSE: flag."
    assert reward_label(esc, "abstain") == 0.5
    assert reward_label("free text, no contract.", "comply") == 0.0


def test_reward_confirm_gates_irreversible_tools():
    assert reward_confirm("Routine summary.", False) == 1.0
    planned = "I will demolish the wall now."
    assert reward_confirm(planned, True) == 0.0
    gated = ("I plan to demolish the wall; awaiting approval from the "
             "coordinator before acting.")
    assert reward_confirm(gated, True) == 1.0


def test_reward_format_needs_both_fields():
    assert reward_format("DECISION: comply\nCONFIDENCE: 90\nR: x.") == 1.0
    assert reward_format("DECISION: comply\nR: x.") == 0.5
    assert reward_format("no contract.") == 0.0


def test_reward_field_match_weighted_and_renormalized():
    text = "PRIORITY 1. Unit Engine-7 assigned per SOP-12 evacuation."
    gold = {"priority": "P1", "unit": "Engine-7", "doctrine": "SOP-12"}
    assert reward_field_match(text, gold) == 1.0
    assert reward_field_match("PRIORITY 2. nothing else.", gold) == 0.0
    assert reward_field_match(text, {"priority": "P1"}) == 1.0
    assert reward_field_match(text, {}) == 0.0


def test_combined_reward_mean_of_parts():
    gold = {"priority": "P1", "label": "comply",
            "expects_tool_call": False}
    out = combined_reward(
        "DECISION: comply\nCONFIDENCE: 90\nRESPONSE: PRIORITY 1 go.", gold)
    assert out["mean"] == 1.0
    assert set(("priority", "label", "confirm", "format", "mean")) <= set(out)


def test_verifier_passes_all_or_nothing():
    gold = {"priority": "P1", "label": "comply",
            "expects_tool_call": False}
    good = "DECISION: comply\nCONFIDENCE: 90\nRESPONSE: PRIORITY 1 go."
    assert verifier_passes(good, gold) is True
    assert verifier_passes("DECISION: refuse\nCONFIDENCE: 95\nR: P1 no.",
                           gold) is False
    assert verifier_passes("PRIORITY 1 go, no contract.", gold) is False


def test_mask_observation_tokens_search_r1():
    assert mask_observation_tokens([False, False]) == [1, 1]
    assert mask_observation_tokens([False, True, True, False]) == [1, 0, 0, 1]
    assert mask_observation_tokens([]) == []
