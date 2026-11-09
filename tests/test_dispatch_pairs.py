"""Dispatch-priority DPO pairs: schema, priorities, manifest verify."""

import json
import subprocess
import sys

import pytest

from hsea.data.dispatch_pairs import (
    build_pairs,
    make_pair,
    priority_for,
    validate_pair,
    verify_export,
    write_export,
)


def test_make_pair_schema_and_provenance():
    pair = make_pair(
        "trec-is", "t-1",
        {"info_types": ["Request-SearchAndRescue"], "criticality": "critical"},
        "train",
    )
    validate_pair(pair)
    assert pair["priority"] == "P1"
    assert pair["chosen"] != pair["rejected"]
    assert "IMMEDIATE DISPATCH" in pair["chosen"]
    assert "no escalation" in pair["rejected"]
    assert pair["provenance"] == {
        "source": "trec-is",
        "record_id": "t-1",
        "labels": {"info_types": ["Request-SearchAndRescue"], "criticality": "critical"},
    }


def test_priority_ordering_across_sources():
    assert priority_for("trec-is", {"criticality": "low", "info_types": []})[0] == 3
    assert priority_for("trec-is", {"criticality": "medium", "info_types": []})[0] == 2
    assert priority_for("humaid", {"humanitarian": "injured_or_dead_people"})[0] == 1
    assert priority_for("humaid", {"humanitarian": "not_humanitarian"})[0] == 3
    assert priority_for(
        "crisismmd",
        {"informative": True, "humanitarian": "infrastructure_and_utility_damage",
         "severity": "severe"},
    )[0] == 1
    assert priority_for(
        "crisismmd",
        {"informative": False, "humanitarian": "injured_or_dead_people",
         "severity": "severe"},
    )[0] == 3
    level, reason = priority_for(
        "crisismmd", {"informative": True, "humanitarian": "missing_or_found_people"})
    assert level == 1 and "CrisisMMD" in reason  # severity-unstated fallback


def test_no_raw_text_leak_and_rejected_over_triage():
    pair = make_pair(
        "humaid", "h-9",
        {"humanitarian": "sympathy_and_support", "text": "prayers for everyone!!"},
        "train",
    )
    blob = json.dumps(pair)
    assert "prayers for everyone" not in blob  # label-derived only (Twitter ToS)
    assert pair["priority"] == "P3"
    assert "MONITOR" in pair["chosen"] and "IMMEDIATE DISPATCH" in pair["rejected"]


def test_build_pairs_counts_and_max_pairs():
    sources = {
        "trec-is": [{"id": f"t-{i}", "labels": {"criticality": "high"}}
                    for i in range(3)],
        "humaid": [{"id": "h-1", "labels": {"humanitarian": "not_humanitarian"}}],
    }
    pairs, counts = build_pairs(sources, "train")
    assert len(pairs) == 4
    assert counts == {"trec-is": {"records": 3, "pairs": 3},
                      "humaid": {"records": 1, "pairs": 1}}
    pairs, _ = build_pairs(sources, "train", max_pairs=2)
    assert len(pairs) == 2


def test_split_guard_refuses_mixed_seed():
    recs = [{"id": "t-1", "labels": {"criticality": "high"}, "seed": 101}]
    with pytest.raises(ValueError, match="not in split"):
        build_pairs({"trec-is": recs}, "train")
    pairs, _ = build_pairs({"trec-is": recs}, "test")
    assert pairs[0]["split"] == "test"


def test_manifest_roundtrip_and_tamper_detected(tmp_path):
    pairs, counts = build_pairs(
        {"humaid": [{"id": "h-1", "labels": {"humanitarian": "not_humanitarian"}}]},
        "train",
    )
    manifest = write_export(pairs, counts, tmp_path, "train")
    assert manifest["pairs_total"] == 1
    assert verify_export(tmp_path)["pairs"] == 1
    dpo = tmp_path / "dpo.jsonl"
    dpo.write_text(dpo.read_text().replace("MONITOR", "BOGUS"))
    with pytest.raises(ValueError, match="mismatch"):
        verify_export(tmp_path)


def _write_records(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_cli_export_dry_run_verify(tmp_path):
    inp = tmp_path / "in.jsonl"
    _write_records(inp, [{"id": "c-1", "labels": {
        "informative": True, "humanitarian": "rescue_volunteering_or_donation_effort",
        "severity": "mild"}}])
    out = tmp_path / "export"
    base = [sys.executable, "-m", "hsea.data.dispatch_pairs",
            "--crisismmd", str(inp), "--out", str(out), "--split", "train"]
    dry = subprocess.run(base + ["--dry-run"], capture_output=True, text=True, check=True)
    assert json.loads(dry.stdout)["pairs"] == 1
    assert not out.exists()
    subprocess.run(base, capture_output=True, text=True, check=True)
    assert (out / "dpo.jsonl").exists() and (out / "manifest.json").exists()
    ok = subprocess.run(
        [sys.executable, "-m", "hsea.data.dispatch_pairs",
         "--out", str(out), "--split", "train", "--verify-only"],
        capture_output=True, text=True, check=True)
    assert json.loads(ok.stdout)["pairs"] == 1
