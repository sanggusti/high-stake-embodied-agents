"""Tests for issue-8 judge analysis (pure, no API)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue8_judge as ij
from judge_pilot import JUDGE_A, JUDGE_B


def _judgment(verdict: str, score: int = 5) -> str:
    return (f'EVIDENCE: "the deciding span of text"\n'
            f'REASONING: The response matches the rubric case exactly, '
            f'so the verdict follows directly from the evidence quoted.\n'
            f'VERDICT: {verdict}\nSCORE: {score}\nCONFIDENCE: 90')


def _rec(template="T1", model="base", task="floodnet_vqa", rid="r1"):
    return {"template": template, "model": model, "task": task, "id": rid}


def test_analyze_agreement_and_rollup():
    inputs = [_rec(rid="r1"), _rec(rid="r2", model="ft"),
              _rec(rid="r3")]
    judgments = {
        "T1:base:floodnet_vqa:r1": {JUDGE_A: _judgment("correct"),
                                    JUDGE_B: _judgment("correct")},
        "T1:ft:floodnet_vqa:r2": {JUDGE_A: _judgment("wrong", 1),
                                  JUDGE_B: _judgment("close", 3)},
        "T1:base:floodnet_vqa:r3": {JUDGE_A: _judgment("correct"),
                                    JUDGE_B: "gibberish"},
    }
    out = ij.analyze_inputs(inputs, judgments)["T1"]
    assert out["n"] == 3
    assert out["parse_fail"] == 1
    assert out["agreement"] == 1 / 3
    assert out["kappa"] is not None
    assert out["by_model"]["base"]["n"] == 1
    assert out["by_model"]["ft"]["verdict_counts"] == {"wrong": 1}


def test_analyze_empty_template_missing():
    assert ij.analyze_inputs([], {}) == {}


def test_analyze_requires_reasoning():
    inputs = [_rec()]
    judgments = {"T1:base:floodnet_vqa:r1": {
        JUDGE_A: 'EVIDENCE: "x"\nREASONING: too short\n'
                 "VERDICT: correct\nSCORE: 5\nCONFIDENCE: 90",
        JUDGE_B: _judgment("correct")}}
    out = ij.analyze_inputs(inputs, judgments)["T1"]
    assert out["parse_fail"] == 1
    assert out["by_model"] == {}
