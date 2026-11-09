"""Flywheel-v2 curation: clearance gate, row shapers, manifest verify."""

import io
import json
import subprocess
import sys

import pytest
from PIL import Image

from hsea.data.clearance import assert_cleared_for_training, disposition
from hsea.data.curate_v2 import (
    V2_SOURCES,
    carry_v1,
    crisitext_row_to_samples,
    disasterm3_row_to_sample,
    floodnet_row_to_sample,
    humaid_row_to_record,
    join_crisismmd,
    ladi_row_to_samples,
    validate_rlvr,
    validate_sft,
    verify_export,
    write_export,
)
from hsea.data.dispatch_pairs import make_pair
from hsea.data.footage import LADI_QUESTIONS


def _png_bytes(size=(8, 6), color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_clearance_clears_v2_sources():
    assert_cleared_for_training(list(V2_SOURCES))
    assert disposition("disasterm3")[0] == "cleared"


def test_clearance_refuses_skipped_blocked_evalonly_unknown():
    for source in ("trec-is", "vidi", "sen1floods11", "crisishelpoffer",
                   "rescuenet", "sard", "flame-3", "heridal", "dad",
                   "not-a-source"):
        with pytest.raises(ValueError, match="(?i)refus"):
            assert_cleared_for_training(["ladi-v2", source])


def test_floodnet_row_shape_refs_only():
    row = {
        "0.img": _png_bytes(),
        "json": {
            "conversations": [
                {"value": "<image>\nIs the area flooded?"},
                {"value": "no"},
            ],
            "meta": {"image_id": "1.JPG", "question_id": 2,
                     "question_type": "Simple_Counting"},
        },
    }
    sample = floodnet_row_to_sample(row, 3)
    validate_sft(sample)
    assert sample["source"] == "floodnet-vqa"
    assert sample["messages"][0]["content"].endswith("[frame omitted]")
    assert sample["messages"][1]["content"] == "no"
    assert sample["provenance"]["stream_index"] == 3
    assert len(sample["provenance"]["image_sha256"]) == 64
    assert "data:image" not in json.dumps(sample)  # refs only, no pixels


def test_ladi_row_renders_12_yes_no():
    row = {"image": Image.open(io.BytesIO(_png_bytes()))}
    for column, _q, _k in LADI_QUESTIONS:
        row[column] = column.endswith("any")
    samples = ladi_row_to_samples(row, 0)
    assert len(samples) == 12
    for sample in samples:
        validate_sft(sample)
        assert sample["messages"][1]["content"] in ("yes", "no")
    assert samples[0]["provenance"]["column"] == "bridges_any"


def test_ladi_row_missing_column_fails_fast():
    with pytest.raises(ValueError, match="missing column"):
        ladi_row_to_samples({"image": Image.open(io.BytesIO(_png_bytes()))}, 0)


def test_disasterm3_row_shape_and_referring_excluded():
    row = {
        "pre_image_path": "train_images\\a_pre.png",
        "post_image_path": "train_images\\a_post.png",
        "post_image_type": "Optical",
        "prompts": "What is damaged?",
        "options_str": "A. Buildings, B. Roads.",
        "training_answer": "Buildings and roads are damaged.",
        "task": "Disaster Bearing Bodies Recognition",
    }
    sample = disasterm3_row_to_sample(row, 0)
    validate_sft(sample)
    assert "Options: A. Buildings" in sample["messages"][0]["content"]
    assert sample["provenance"]["post_image_type"] == "Optical"
    assert disasterm3_row_to_sample({**row, "task": "Referring Expression Segmentation"}, 1) is None
    with pytest.raises(ValueError, match="lacks prompt"):
        disasterm3_row_to_sample({"task": "x"}, 2)


def test_crisitext_row_per_message_and_synthetic_skipped():
    row = {
        "scenario_id": "train-0",
        "source": "FEMA",
        "guidelines": "['winterweather']",
        "original_description": "BULLETIN TEXT",
        "messages": (
            "[{'event_id': 'train-0-0', 'event': 'Snow coming.', "
            "'message': 'Prepare for snow.'}]"
        ),
    }
    samples = crisitext_row_to_samples(row, 0)
    assert len(samples) == 1
    validate_sft(samples[0])
    assert "FEMA guidelines: winterweather" in samples[0]["messages"][0]["content"]
    assert samples[0]["messages"][1]["content"] == "Prepare for snow."
    assert crisitext_row_to_samples({**row, "source": "generated"}, 1) == []
    with pytest.raises(ValueError, match="unparsable"):
        crisitext_row_to_samples({**row, "messages": "nope["}, 2)


def test_humaid_record_labels_only():
    rec = humaid_row_to_record(
        {"tweet_text": "secret post text", "class_label": "not_humanitarian"}, 7)
    assert rec == {"id": "humaid-train-000007",
                   "labels": {"humanitarian": "not_humanitarian"}}
    assert "secret post text" not in json.dumps(rec)


def test_join_crisismmd_maps_severity_and_drops():
    hum = {"t1": "injured_or_dead_people", "t2": "not_humanitarian",
           "t3": "other_relevant_information"}
    info = {"t1": "informative", "t2": "not_informative"}
    dmg = {"t1": "severe_damage"}
    records, stats = join_crisismmd(hum, info, dmg, ["t1", "t2", "t3"])
    assert stats == {"tweets_seen": 3, "joined": 2, "dropped": 1}
    assert records[0]["labels"] == {
        "informative": True, "humanitarian": "injured_or_dead_people",
        "severity": "severe",
    }
    assert records[1]["labels"]["informative"] is False


def _sft_sample():
    return {
        "messages": [
            {"role": "user", "content": "q\n[frame omitted]"},
            {"role": "assistant", "content": "a"},
        ],
        "source": "ladi-v2",
        "provenance": {"revision": "abc"},
    }


def _rlvr_record():
    return {
        "sample_id": "s-1", "scenario": "forest_fire", "seed": 1,
        "world_hash": "w", "rewards": {"composite": 0.6}, "composite": 0.6,
        "teacher": "t", "log": "l.eval", "l2": None, "source": "l2",
        "provenance": {"origin": "logs"},
    }


def test_manifest_roundtrip_and_tamper_detected(tmp_path):
    dpo = make_pair("humaid", "h-1",
                    {"humanitarian": "not_humanitarian"}, "train")
    sources_meta = {s: {"license": "x", "stats": {}} for s in V2_SOURCES}
    manifest = write_export([_sft_sample()], [dpo], [_rlvr_record()],
                            sources_meta, tmp_path, "train", "cmd", {})
    assert manifest["totals"] == {"sft": 1, "dpo": 1, "rlvr": 1}
    assert verify_export(tmp_path)["rows"] == {"sft": 1, "dpo": 1, "rlvr": 1}
    sft = tmp_path / "sft.jsonl"
    sft.write_text(sft.read_text().replace("[frame omitted]", "BOGUS"))
    with pytest.raises(ValueError, match="mismatch"):
        verify_export(tmp_path)


def test_verify_rejects_bad_source_and_schema(tmp_path):
    bad = _sft_sample()
    bad["source"] = "trec-is"
    sources_meta = {s: {"license": "x", "stats": {}} for s in V2_SOURCES}
    with pytest.raises(ValueError, match="bad source"):
        write_export([bad], [], [], sources_meta, tmp_path, "train", "cmd", {})
    with pytest.raises(ValueError, match="rewards dict"):
        validate_rlvr({**_rlvr_record(), "rewards": []})


def test_carry_v1_stamps_origin(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "manifest.json").write_text(json.dumps(
        {"teacher": "Q", "code_commit": "c"}))
    (v1 / "sft.jsonl").write_text(json.dumps(
        {"messages": [{"role": "user", "content": "q"}],
         "sample_id": "s", "seed": 1}) + "\n")
    (v1 / "rlvr.jsonl").write_text(json.dumps(
        {"sample_id": "s", "scenario": "f", "seed": 1, "world_hash": "w",
         "rewards": {}, "composite": 0.1, "l2": "x"}) + "\n")
    sft, rlvr, stats = carry_v1(v1)
    assert stats == {"sft_pairs": 1, "episodes": 1, "teacher": "Q", "commit": "c"}
    assert sft[0]["provenance"]["origin"] == "flywheel-v1"
    assert rlvr[0]["provenance"]["origin"] == "flywheel-v1"
    validate_sft(sft[0])
    validate_rlvr(rlvr[0])


def test_cli_dry_run_and_verify(tmp_path):
    out = tmp_path / "export"
    base = [sys.executable, "-m", "hsea.data.curate_v2",
            "--out", str(out), "--split", "train"]
    dry = subprocess.run(base + ["--dry-run"], capture_output=True, text=True,
                         check=True)
    assert json.loads(dry.stdout)["hf_dataset"].endswith("dmb-flywheel-v2")
    assert not out.exists()
    dpo = make_pair("crisismmd", "c-1",
                    {"informative": True, "humanitarian": "other_relevant_information"},
                    "train")
    sources_meta = {s: {"license": "x", "stats": {}} for s in V2_SOURCES}
    write_export([_sft_sample()], [dpo], [_rlvr_record()],
                 sources_meta, out, "train", "cmd", {})
    ok = subprocess.run(base + ["--verify-only"], capture_output=True, text=True,
                        check=True)
    assert json.loads(ok.stdout)["rows"]["dpo"] == 1
