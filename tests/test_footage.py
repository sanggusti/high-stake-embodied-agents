"""Footage scoring + sample building (no model calls, no downloads)."""

import base64
import io

from PIL import Image

from evals.floodnet_vqa import floodnet_vqa as _floodnet_task  # noqa: F401
from evals.ladi_v2 import ladi_v2 as _ladi_task  # noqa: F401
from hsea.data.footage import (
    LADI_QUESTIONS,
    footage_sample,
    image_data_url,
    png_bytes,
    question_kind,
)
from hsea.scorers.footage import (
    compute_counting,
    compute_footage,
    compute_grounding,
    compute_vqa,
    extract_int,
    normalize_answer,
)


def _truth(kind="vqa", answer="yes"):
    return {"kind": kind, "answer": answer}


def test_normalize_and_vqa():
    assert normalize_answer("  Yes! ") == "yes"
    assert normalize_answer("Non-flooded.") == "nonflooded"
    assert compute_vqa("YES.", "yes") == 1.0
    assert compute_vqa("no", "yes") == 0.0
    assert compute_vqa("", "yes") == 0.0
    assert compute_vqa(None, "yes") == 0.0


def test_counting_exact_and_mae():
    assert compute_counting("4", "4") == (1.0, 0.0)
    assert compute_counting("There are 4 buildings.", "4") == (1.0, 0.0)
    assert compute_counting("6", "4") == (0.0, 2.0)
    acc, mae = compute_counting("many", "4")
    assert acc == 0.0 and mae is None
    assert extract_int("no digits") is None
    assert extract_int("12abc34") == 12


def test_grounding_cell_match():
    assert compute_grounding("The damage is at B3.", "b3") == 1.0
    assert compute_grounding("B3", "B3") == 1.0
    assert compute_grounding("C4", "B3") == 0.0
    assert compute_grounding("bottom left", "B3") == 0.0
    assert compute_grounding("", "B3") == 0.0


def test_routing_and_totality():
    vqa = compute_footage("yes", _truth("vqa", "yes"))
    assert vqa["vqa_accuracy"] == 1.0 and vqa["footage_composite"] == 1.0
    assert vqa["counting_acc"] is None
    sev = compute_footage("no", _truth("severity", "yes"))
    assert sev["severity_f1"] == 0.0 and sev["footage_composite"] == 0.0
    cnt = compute_footage("5", _truth("counting", "4"))
    assert cnt["counting_acc"] == 0.0 and cnt["counting_mae"] == 1.0
    assert cnt["footage_composite"] == 0.0
    gnd = compute_footage("A1", _truth("grounding", "A1"))
    assert gnd["grounding_acc"] == 1.0
    # Malformed input never raises; unknown kind falls back to VQA.
    assert compute_footage(None, _truth("vqa", "yes"))["footage_composite"] == 0.0
    assert compute_footage("yes", _truth("mystery", "yes"))["vqa_accuracy"] == 1.0


def test_question_kind_routing():
    assert question_kind("Simple_Counting") == "counting"
    assert question_kind("Complex_Counting") == "counting"
    assert question_kind("Density_Estimation") == "vqa"
    assert question_kind("Risk_Assessment") == "vqa"
    assert question_kind("Road_Condition_Recognition") == "vqa"


def test_ladi_questions_cover_severity():
    kinds = {kind for _, _, kind in LADI_QUESTIONS}
    assert kinds == {"vqa", "severity"}
    assert len(LADI_QUESTIONS) == 12
    assert sum(1 for _, _, k in LADI_QUESTIONS if k == "severity") == 6


def _red_dot():
    img = Image.new("RGB", (16, 16), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_resize_and_data_url():
    big = Image.new("RGB", (3000, 4000), (0, 128, 0))
    encoded = png_bytes(big, max_side=1024)
    small = Image.open(io.BytesIO(encoded))
    assert max(small.size) == 1024
    url, digest = image_data_url(_red_dot())
    assert url.startswith("data:image/png;base64,")
    assert len(digest) == 64
    payload = base64.b64decode(url.split(",", 1)[1])
    assert Image.open(io.BytesIO(payload)).size == (16, 16)
    # Deterministic: same bytes -> same digest.
    assert image_data_url(_red_dot())[1] == digest


def test_footage_sample_hides_truth():
    url, digest = image_data_url(_red_dot())
    first = footage_sample("s-1", "Is it flooded?", "yes", "vqa", url, digest, {"corpus": "t"})
    second = footage_sample("s-1", "Is it flooded?", "yes", "vqa", url, digest, {"corpus": "t"})
    assert first.id == second.id == "s-1"
    assert first.metadata["answer"] == "yes"
    assert first.metadata["image_sha256"] == digest
    assert first.target == "yes"
    [msg] = first.input
    kinds = {type(part).__name__ for part in msg.content}
    assert kinds == {"ContentText", "ContentImage"}
    assert "yes" not in msg.content[0].text  # question carries no answer


def test_tasks_importable():
    assert callable(_floodnet_task) and callable(_ladi_task)
