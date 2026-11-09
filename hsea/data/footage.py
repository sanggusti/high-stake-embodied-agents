"""Footage-grounded VLM eval harness (#19): deterministic scoring on real corpora.

Corpus 1: FloodNet-VQA via ``ljx620/FloodNet-VQA`` (ungated webdataset mirror;
native UAV VQA pairs). Corpus 2: LADI v2 via ``MITLL/LADI-v2-dataset``
(CC-BY-4.0; expert multi-labels rendered as yes/no QA pairs).

Truth answers stay in ``Sample.metadata`` (hidden from the model); images are
streamed from the pinned HF revision and embedded as resized PNG data URLs.
Images are never committed — only ``image_sha256`` fingerprints land in logs.
"""

from __future__ import annotations

import hashlib
import io
import os

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from PIL import Image

from hsea.perception.frames import to_data_url

FLOODNET_VQA_ID = "ljx620/FloodNet-VQA"
FLOODNET_VQA_REVISION = "ec9277afd48ffb27b945eac3c6fb8d8aa77f8cfd"
FLOODNET_VQA_SPLIT = "validation"
# Mirror card carries no license tag (verified 2026-11-09); annotations derive
# from FloodNet (CDLA-Permissive-1.0). Eval use only until clarified.

LADI_V2_ID = "MITLL/LADI-v2-dataset"
LADI_V2_REVISION = "5f2dbfe8c466d32edafd1bab847ec5252309acdb"
LADI_V2_SPLIT = "test"  # CC-BY-4.0, official MITLL upload.

COUNTING_TYPES = frozenset({"Simple_Counting", "Complex_Counting"})

FOOTAGE_SYSTEM = (
    "You are a disaster-imagery analyst. Answer the question about the "
    "image using a single word or phrase."
)


def question_kind(question_type: str) -> str:
    """Route a FloodNet question type to its scoring metric."""
    return "counting" if question_type in COUNTING_TYPES else "vqa"


# (LADI column, question, scoring kind). Damage-grade columns route to
# severity; presence columns route to plain VQA accuracy.
LADI_QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("bridges_any", "Is there a bridge in this image?", "vqa"),
    ("buildings_any", "Are there any buildings in this image?", "vqa"),
    ("buildings_affected_or_greater", "Are any buildings affected or worse?", "severity"),
    ("buildings_minor_or_greater", "Do any buildings show at least minor damage?", "severity"),
    ("debris_any", "Is there any debris in this image?", "severity"),
    ("flooding_any", "Is there any flooding in this image?", "vqa"),
    ("flooding_structures", "Are any structures flooded?", "severity"),
    ("roads_any", "Are there any roads in this image?", "vqa"),
    ("roads_damage", "Do any roads show damage?", "severity"),
    ("trees_any", "Are there any trees in this image?", "vqa"),
    ("trees_damage", "Do any trees show damage?", "severity"),
    ("water_any", "Is there any water in this image?", "vqa"),
)


def png_bytes(image: Image.Image, max_side: int = 1024) -> bytes:
    """Resize to ``max_side`` (long edge) and encode PNG. Deterministic."""
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_data_url(raw: bytes, max_side: int = 1024) -> tuple[str, str]:
    """``raw`` image bytes -> (PNG data URL, sha256 of encoded PNG)."""
    encoded = png_bytes(Image.open(io.BytesIO(raw)), max_side=max_side)
    return to_data_url(encoded), hashlib.sha256(encoded).hexdigest()


def footage_sample(
    sample_id: str,
    question: str,
    answer: str,
    kind: str,
    data_url: str,
    image_sha256: str,
    extra: dict,
) -> Sample:
    """Multimodal sample with truth hidden in metadata (cf. recon_advisor)."""
    metadata = {
        "sample_id": sample_id,
        "answer": answer,
        "kind": kind,
        "image_sha256": image_sha256,
        **extra,
    }
    return Sample(
        id=sample_id,
        input=[
            ChatMessageUser(
                content=[
                    ContentText(text=question),
                    ContentImage(image=data_url, detail="auto"),
                ]
            )
        ],
        target=answer,
        metadata=metadata,
    )


def _limit_from_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def build_floodnet_samples(
    limit: int | None = None, split: str = FLOODNET_VQA_SPLIT
) -> list[Sample]:
    """First ``limit`` VQA pairs of the pinned mirror, in stream order."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, eval-time only)

    cap = limit if limit is not None else _limit_from_env("HSEA_FOOTAGE_LIMIT", 25)
    ds = load_dataset(
        FLOODNET_VQA_ID, split=split, revision=FLOODNET_VQA_REVISION, streaming=True
    )
    samples = []
    for i, row in enumerate(ds):
        if i >= cap:
            break
        convo = row["json"]["conversations"]
        question = convo[0]["value"].replace("<image>\n", "")
        answer = convo[1]["value"]
        meta = row["json"]["meta"]
        qtype = meta.get("question_type", "")
        data_url, digest = image_data_url(bytes(row["0.img"]))
        samples.append(
            footage_sample(
                sample_id=f"floodnet-{split}-{i:04d}-q{meta.get('question_id', i)}",
                question=question,
                answer=answer,
                kind=question_kind(qtype),
                data_url=data_url,
                image_sha256=digest,
                extra={
                    "corpus": "floodnet-vqa",
                    "split": split,
                    "revision": FLOODNET_VQA_REVISION,
                    "image_id": meta.get("image_id", ""),
                    "question_type": qtype,
                },
            )
        )
    return samples


def build_ladi_samples(
    max_images: int | None = None, split: str = LADI_V2_SPLIT
) -> list[Sample]:
    """All :data:`LADI_QUESTIONS` per image for the first ``max_images``."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, eval-time only)

    cap = (
        max_images
        if max_images is not None
        else _limit_from_env("HSEA_FOOTAGE_MAX_IMAGES", 10)
    )
    ds = load_dataset(
        LADI_V2_ID, split=split, revision=LADI_V2_REVISION, streaming=True
    )
    samples = []
    for i, row in enumerate(ds):
        if i >= cap:
            break
        buf = io.BytesIO()
        row["image"].save(buf, format="PNG")
        data_url, digest = image_data_url(buf.getvalue())
        for column, question, kind in LADI_QUESTIONS:
            answer = "yes" if row[column] else "no"
            samples.append(
                footage_sample(
                    sample_id=f"ladi-{split}-{i:04d}-{column}",
                    question=question,
                    answer=answer,
                    kind=kind,
                    data_url=data_url,
                    image_sha256=digest,
                    extra={
                        "corpus": "ladi-v2",
                        "split": split,
                        "revision": LADI_V2_REVISION,
                        "column": column,
                    },
                )
            )
    return samples
