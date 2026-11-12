"""Repair mix pins (#73 Stage 4): counts, gate discipline, KTO finalize."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compose_aya73 import split_safety  # noqa: E402
from compose_aya73_repair import finalize_kto  # noqa: E402

DATA = ROOT / "data" / "aya73"


def _rows(name: str) -> list[dict]:
    with open(DATA / name) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_repair_mix_counts():
    manifest = json.loads((DATA / "repair_manifest.json").read_text())
    assert manifest["seed"] == 73
    assert manifest["sft"]["total"] == 334
    assert manifest["sft"] == {"safety": 26, "abstention_cal": 28,
                              "dispatch_anchors": 200, "wj_method": 80,
                              "total": 334}
    assert manifest["dpo"] == {"safety": 26, "dispatch_debiased": 500,
                              "total": 526}
    assert len(_rows("repair_sft.jsonl")) == 334
    assert len(_rows("repair_dpo.jsonl")) == 526
    assert len(_rows("kto.jsonl")) >= 56  # drift negatives append later


def test_gate_benches_never_train():
    from hsea.safety.data import ITEMS

    gate_prompts = {p for b, _c, _l, p in ITEMS
                    if b in ("harmbench", "airbench")}
    assert len(gate_prompts) == 26
    train_prompts = set()
    for rec in _rows("repair_sft.jsonl"):
        for msg in rec["messages"]:
            if msg["role"] == "user":
                train_prompts.add(msg["content"])
    for rec in _rows("repair_dpo.jsonl"):
        train_prompts.add(rec["prompt"])
    assert not (gate_prompts & train_prompts)
    # The 26 safety demos are exactly the Phase-1 non-gate train half.
    train_items, _ = split_safety()
    assert len(train_items) == 26


def test_abstention_eval_split_never_trains():
    eval_ids = set()
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for line in open(ROOT / "data" / "abstention" / name):
            rec = json.loads(line)
            if rec["split"] == "eval":
                eval_ids.add(rec["id"])
    assert len(eval_ids) == 52
    for rec in _rows("repair_sft.jsonl"):
        prov = rec.get("provenance", {})
        if prov.get("mix") == "aya73-repair-sft" and "id" in prov \
                and rec["source"] == "hsea-abstention":
            assert prov["id"] not in eval_ids
    for rec in _rows("kto.jsonl"):
        assert rec["provenance"]["id"] not in eval_ids


def test_abstention_keeps_action_footer():
    for rec in _rows("repair_sft.jsonl"):
        if rec["source"] != "hsea-abstention":
            continue
        text = rec["messages"][-1]["content"]
        assert text.count("ACTION:") == 1  # single-verdict discipline
        assert "DECISION:" not in text  # no #74-style migration
    for rec in _rows("kto.jsonl"):
        if rec["provenance"]["side"] == "desirable":
            assert rec["completion"].count("ACTION:") == 1


def test_repair_dpo_debiased():
    for rec in _rows("repair_dpo.jsonl"):
        if rec["source"] != "flywheel-dispatch-debiased":
            continue
        assert not rec["chosen"].startswith("PRIORITY")
        assert not rec["rejected"].startswith("PRIORITY")
        assert rec["chosen"] and rec["rejected"]
        assert rec["system"]  # DPO mapper needs system + prompt


def test_repair_disjoint_from_held_out():
    xs = {json.loads(line)["prompt"] for line in
          open(ROOT / "data" / "xstest" / "lookalikes.jsonl")}
    train = set()
    for rec in _rows("repair_sft.jsonl"):
        for msg in rec["messages"]:
            if msg["role"] == "user":
                train.add(msg["content"])
    for rec in _rows("repair_dpo.jsonl"):
        train.add(rec["prompt"])
    assert not (xs & train)
    held = {json.loads(line)["situation"] for line in
            open(ROOT / "data" / "dispatch-pairs" / "dpo.jsonl")}
    assert not (held & train)


def test_kto_shape():
    for rec in _rows("kto.jsonl"):
        assert set(("prompt", "completion", "label")) <= set(rec)
        assert isinstance(rec["label"], bool)


def test_finalize_kto_mines_drift(tmp_path):
    comps = tmp_path / "eval_dpo.jsonl"
    cal = [json.loads(line) for line in
           open(ROOT / "data" / "abstention" / "scenarios.jsonl")
           if json.loads(line)["split"] == "cal"]
    first, second = cal[0], cal[1]
    drift = ("Some reasoning.\nACTION: comply\nCONFIDENCE: 70\nMore.\n"
             f"ACTION: {first['label']}\nCONFIDENCE: 60\nEven more.\n"
             "ACTION: escalate\nCONFIDENCE: 55")
    single = f"Clean reasoning.\nACTION: {second['label']}\nCONFIDENCE: 90"
    # Force the drift case to actually miss gold at the last verdict.
    if first["label"] == "escalate":
        drift = drift.replace("ACTION: escalate\nCONFIDENCE: 55",
                              "ACTION: comply\nCONFIDENCE: 55")
    with open(comps, "w") as fh:
        fh.write(json.dumps({"key": f"abstention:{first['id']}",
                             "completion": drift}) + "\n")
        fh.write(json.dumps({"key": f"abstention:{second['id']}",
                             "completion": single}) + "\n")
    drift_rows, n_seen = finalize_kto(comps)
    assert n_seen == 2
    assert len(drift_rows) == 1
    row = drift_rows[0]
    assert row["label"] is False
    assert row["provenance"]["id"] == first["id"]
    assert row["provenance"]["violation"] == "dpo-verdict-drift"
    assert row["completion"] == drift
