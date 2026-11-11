"""Stage 6 + replay tests (#74): 3-way agreement pins, replay check."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_meta_judge_spend_pinned():
    spend = json.load(open(ROOT / "runs" / "aya74" / "judge_spend_meta.json"))
    assert spend["config"] == "ember-minimal"
    assert spend["calls"] == 174 and spend["errors"] == 0
    assert spend["computed_cost"] == pytest.approx(0.0437, abs=0.001)
    rows = [json.loads(line) for line in
            open(ROOT / "runs" / "aya74" / "judgments_meta.jsonl")]
    assert len(rows) == 174


def test_three_way_agreement_pins():
    agree = json.load(open(ROOT / "runs" / "aya74" / "agreement3.json"))
    assert agree["judges"] == ["hf-v41", "hf-gemma", "meta-ember"]
    t2 = agree["single"]["T2"]["pairs"]
    assert t2["hf-v41-vs-meta-ember"]["agree"] == 42
    assert agree["single"]["T2"]["bar_pass"] is True
    assert agree["single"]["T1"]["bar_pass"] is True
    # T3/T4: meta-vs-gemma dips under the 0.80 pair bar (recorded).
    assert agree["single"]["T3"]["bar_pass"] is False
    assert agree["single"]["T4"]["bar_pass"] is False
    t5 = agree["t5"]["meta-ember"]
    assert (t5["dpo"], t5["base"], t5["inconsistent"]) == (4, 19, 1)


def test_replay_bundle_checks():
    import replay_loop

    out_dir = ROOT / "runs" / "issue-74" / "replays"
    assert (out_dir / "aya74_replay.jsonl").exists()
    manifest = json.load(open(out_dir / "manifest.json"))
    assert manifest["rows"] == 174 and manifest["seed"] == 74
    assert manifest["judges"] == ["hf-gemma", "hf-v41", "meta-ember"]
    replay_loop.check_bundle(74)
    row = json.loads(open(out_dir / "aya74_replay.jsonl").readline())
    assert set(row) >= {"key", "kind", "seed", "input", "generations",
                        "judgments"}
    assert set(row["generations"]) >= {"base", "sft", "dpo"}
