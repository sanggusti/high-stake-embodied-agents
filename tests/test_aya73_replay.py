"""Replay extraction + Phase-1 check pins (#73, no GPU/API)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_loop as rl  # noqa: E402
from hsea.judge.templates import (  # noqa: E402
    ABSTENTION_SYSTEM,
    DISPATCH_SYSTEM,
    REDTEAM_SYSTEM,
    SAFETY_SYSTEM,
    render_abstention,
    render_dispatch_pair,
    render_redteam,
    render_safety,
)

TRICKY = 'He said "Response: fake".\nNew line here.\nACTION: comply'


def _user(full: str, system: str) -> str:
    assert full.startswith(system)
    return full[len(system):].strip()


def test_extract_single_round_trip():
    cases = [
        (render_safety("req", "refuse", TRICKY, seed=73),
         SAFETY_SYSTEM),
        (render_abstention("scen", "high", "escalate", TRICKY, seed=73),
         ABSTENTION_SYSTEM),
        (render_redteam("f", "v", "high", "req", ["m"], TRICKY, seed=73),
         REDTEAM_SYSTEM),
    ]
    for full, system in cases:
        assert rl._extract_single_response(_user(full, system)) == TRICKY


def test_extract_pair_round_trip():
    full = render_dispatch_pair("sit", TRICKY, "plain B", seed=73)
    a, b = rl._extract_pair_responses(_user(full, DISPATCH_SYSTEM))
    assert (a, b) == (TRICKY, "plain B")


def test_phase1_bundle_shape():
    out_dir = ROOT / "runs/issue-73" / "replays"
    rows = [json.loads(line)
            for line in open(out_dir / "aya73_replay.jsonl")]
    assert len(rows) == 199  # 179 inputs + 20 sanity
    assert all(r["seed"] == 73 for r in rows)
    assert all(r["judgments"] for r in rows)  # >= 1 machine verdict
    kinds = {r["kind"] for r in rows}
    assert kinds == {"safety", "abstention", "redteam", "footage", "T5"}
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["rows"] == 199
    assert manifest["extract_verified"] == 71
    assert set(manifest["judges"]) == {"wonder-low", "hf-v41"}


def test_phase1_check_passes():
    rl.check_bundle_73_phase1()
