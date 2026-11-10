"""Judge templates: render, parse, and playbook-verbatim checks (#62)."""

import re
from pathlib import Path

import pytest

from hsea.judge import templates as T

PLAYBOOK = Path(__file__).resolve().parents[1] / "docs" / "judge-playbook.md"


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


# --- render ---------------------------------------------------------------

def test_render_safety_holds_all_parts_and_hides_model():
    prompt = T.render_safety("Do X?", "refuse", "I can't do X.")
    assert T.SAFETY_SYSTEM in prompt
    assert T.SAFETY_RUBRIC in prompt
    assert T.OUTPUT_SCHEMA in prompt
    assert "Do X?" in prompt and "I can't do X." in prompt
    for leak in ("muse-spark", "v41", "deepseek", "gemma", "gpt"):
        assert leak not in prompt.lower()


def test_render_abstention_states_footer_rule():
    prompt = T.render_abstention("scenario", "high", "escalate", "resp")
    assert "ACTION footer never" in prompt
    assert "escalate" in prompt


def test_render_footage_kind_aware():
    prompt = T.render_footage("counting", "How many?", "4", "Four")
    assert "counting" in prompt and "How many?" in prompt


def test_render_redteam_carries_markers():
    prompt = T.render_redteam("retrieval_injection", "override", "high",
                              "req", ("sector 7",), "resp")
    assert "sector 7" in prompt


def test_render_dispatch_pair_blind_labels():
    prompt = T.render_dispatch_pair("sit", "AAA", "BBB")
    assert 'Response A: "AAA"' in prompt
    assert 'Response B: "BBB"' in prompt
    assert T.PAIRWISE_SCHEMA in prompt


def test_render_seed_deterministic_and_optional():
    a = T.render_safety("r", "refuse", "x", seed=62)
    b = T.render_safety("r", "refuse", "x", seed=62)
    assert a == b
    c = T.render_safety("r", "refuse", "x", seed=None)
    assert "Example 1 instance" in c


def test_no_em_dash_in_templates():
    for name in ("FOOTAGE_SYSTEM", "SAFETY_SYSTEM", "ABSTENTION_SYSTEM",
                 "REDTEAM_SYSTEM", "DISPATCH_SYSTEM", "FOOTAGE_RUBRIC",
                 "SAFETY_RUBRIC", "ABSTENTION_RUBRIC", "REDTEAM_RUBRIC",
                 "DISPATCH_RUBRIC", "OUTPUT_SCHEMA", "PAIRWISE_SCHEMA"):
        assert "—" not in getattr(T, name), name
    for name in ("FOOTAGE_FEWSHOTS", "SAFETY_FEWSHOTS",
                 "ABSTENTION_FEWSHOTS", "REDTEAM_FEWSHOTS",
                 "DISPATCH_FEWSHOTS"):
        for inst, judge in getattr(T, name):
            assert "—" not in inst + judge, name


# --- parse ----------------------------------------------------------------

def test_parse_all_fewshot_judgments():
    cases = [(T.FOOTAGE_FEWSHOTS, T.FOOTAGE_LABELS),
             (T.SAFETY_FEWSHOTS, T.SAFETY_LABELS),
             (T.ABSTENTION_FEWSHOTS, T.ABSTENTION_LABELS),
             (T.REDTEAM_FEWSHOTS, T.REDTEAM_LABELS)]
    for shots, labels in cases:
        for _, judgment in shots:
            parsed = T.parse_judgment(judgment, labels)
            assert parsed["ok"], parsed["errors"]
    for _, judgment in T.DISPATCH_FEWSHOTS:
        parsed = T.parse_pairwise(judgment)
        assert parsed["ok"], parsed["errors"]


def test_parse_last_marker_wins():
    body = ("EVIDENCE: a\nREASONING: " + "word " * 12
            + "\nVERDICT: comply\nSCORE: 2\nVERDICT: refuse\nSCORE: 5\n"
            + "CONFIDENCE: 90")
    parsed = T.parse_judgment(body, T.SAFETY_LABELS)
    assert parsed["ok"] and parsed["verdict"] == "refuse"
    assert parsed["score"] == 5


def test_parse_failure_modes():
    assert "verdict_missing" in T.parse_judgment("", T.SAFETY_LABELS)["errors"]
    bad_verdict = ("EVIDENCE: a\nREASONING: " + "word " * 12
                   + "\nVERDICT: maybe\nSCORE: 5\nCONFIDENCE: 90")
    parsed = T.parse_judgment(bad_verdict, T.SAFETY_LABELS)
    assert not parsed["ok"] and any(e.startswith("verdict_invalid")
                                    for e in parsed["errors"])
    short = ("EVIDENCE: a\nREASONING: too short here\nVERDICT: refuse\n"
             + "SCORE: 5\nCONFIDENCE: 90")
    assert "reasoning_short" in T.parse_judgment(short,
                                                 T.SAFETY_LABELS)["errors"]
    bad_score = ("EVIDENCE: a\nREASONING: " + "word " * 12
                 + "\nVERDICT: refuse\nSCORE: 9\nCONFIDENCE: 90")
    assert "score_range:9" in T.parse_judgment(bad_score,
                                               T.SAFETY_LABELS)["errors"]
    assert T.parse_judgment(None, T.SAFETY_LABELS)["ok"] is False


def test_parse_pairwise_last_token_wins_and_tie():
    body = ("EVIDENCE: A: x\nREASONING: " + "word " * 12
            + "\nVERDICT: [[A]]\nVERDICT: [[C]]\nMARGIN: 1\nCONFIDENCE: 99")
    parsed = T.parse_pairwise(body)
    assert parsed["ok"] and parsed["verdict"] == "C"


def test_estimate_tokens_ceiling():
    assert T.estimate_tokens("") == 1
    assert T.estimate_tokens("abcd") == 1
    assert T.estimate_tokens("abcde") == 2


def test_canonical_fewshots_rejects_unknown():
    with pytest.raises(ValueError, match="unknown template"):
        T.canonical_fewshots("nope")


# --- playbook verbatim ------------------------------------------------------

_VERBATIM = {
    "OUTPUT_SCHEMA": T.OUTPUT_SCHEMA,
    "PAIRWISE_SCHEMA": T.PAIRWISE_SCHEMA,
    "FOOTAGE_SYSTEM": T.FOOTAGE_SYSTEM,
    "FOOTAGE_RUBRIC": T.FOOTAGE_RUBRIC,
    "FOOTAGE_FEWSHOTS": T.canonical_fewshots("footage"),
    "SAFETY_SYSTEM": T.SAFETY_SYSTEM,
    "SAFETY_RUBRIC": T.SAFETY_RUBRIC,
    "SAFETY_FEWSHOTS": T.canonical_fewshots("safety"),
    "ABSTENTION_SYSTEM": T.ABSTENTION_SYSTEM,
    "ABSTENTION_RUBRIC": T.ABSTENTION_RUBRIC,
    "ABSTENTION_FEWSHOTS": T.canonical_fewshots("abstention"),
    "REDTEAM_SYSTEM": T.REDTEAM_SYSTEM,
    "REDTEAM_RUBRIC": T.REDTEAM_RUBRIC,
    "REDTEAM_FEWSHOTS": T.canonical_fewshots("redteam"),
    "DISPATCH_SYSTEM": T.DISPATCH_SYSTEM,
    "DISPATCH_RUBRIC": T.DISPATCH_RUBRIC,
    "DISPATCH_FEWSHOTS": T.canonical_fewshots("dispatch"),
}


@pytest.mark.parametrize("block", sorted(_VERBATIM))
def test_playbook_embeds_template_verbatim(block):
    body = PLAYBOOK.read_text()
    pattern = re.compile(
        r"<!-- VERBATIM:" + block + r":START -->\n(.*?)"
        r"<!-- VERBATIM:" + block + r":END -->",
        re.DOTALL,
    )
    match = pattern.search(body)
    assert match, f"missing verbatim block {block}"
    captured = match.group(1).strip()
    if captured.startswith("```"):
        captured = "\n".join(captured.splitlines()[1:])
    if captured.rstrip().endswith("```"):
        captured = "\n".join(captured.rstrip().splitlines()[:-1])
    assert _norm(captured) == _norm(_VERBATIM[block]), block
