"""Footage VLM scoring (#19): deterministic QA/counting/severity/grounding.

Each sample carries one question; ``metadata["kind"]`` routes to its metric.
The scorer value is the routed metric (``footage_composite``); the full vector
sits in metadata (cf. ``docs/metrics.md`` convention). All pure functions are
total: malformed model output scores 0, never raises.
"""

from __future__ import annotations

import re
import string

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

_INT_RE = re.compile(r"[-+]?\d+")
_CELL_RE = re.compile(r"\b([A-H])\s*([1-8])\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip, drop punctuation, collapse whitespace."""
    return " ".join((text or "").lower().translate(_PUNCT_TABLE).split())


def compute_vqa(pred: str, gold: str) -> float:
    """Normalized exact match (yes/no + short phrases)."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def extract_int(text: str) -> int | None:
    """First integer in the text, or None when the output has no number."""
    match = _INT_RE.search(text or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def compute_counting(pred: str, gold: str) -> tuple[float, float | None]:
    """(exact-match accuracy, absolute error or None when unparseable)."""
    pred_n, gold_n = extract_int(pred), extract_int(gold)
    if pred_n is None or gold_n is None:
        return 0.0, None
    return (1.0 if pred_n == gold_n else 0.0), float(abs(pred_n - gold_n))


def compute_grounding(pred: str, gold_cell: str) -> float:
    """Grid-cell match (e.g. "B3"): deterministic grounding without boxes."""
    pred_m = _CELL_RE.search(pred or "")
    gold_m = _CELL_RE.search(gold_cell or "")
    if not pred_m or not gold_m:
        return 0.0
    return (
        1.0
        if pred_m.groups()[0].upper() == gold_m.groups()[0].upper()
        and pred_m.groups()[1] == gold_m.groups()[1]
        else 0.0
    )


def compute_footage(pred: str, truth: dict) -> dict[str, float | None]:
    """Route by ``truth["kind"]``; inapplicable metrics stay None."""
    kind = truth.get("kind", "vqa")
    gold = str(truth.get("answer", ""))
    vector: dict[str, float | None] = {
        "vqa_accuracy": None,
        "counting_acc": None,
        "counting_mae": None,
        "severity_f1": None,
        "grounding_acc": None,
    }
    if kind == "counting":
        acc, mae = compute_counting(pred, gold)
        vector["counting_acc"] = acc
        vector["counting_mae"] = mae
        composite = acc
    elif kind == "severity":
        # Single yes/no EM per sample; the mean over samples is micro-F1
        # over the binary damage judgments.
        vector["severity_f1"] = compute_vqa(pred, gold)
        composite = vector["severity_f1"]
    elif kind == "grounding":
        vector["grounding_acc"] = compute_grounding(pred, gold)
        composite = vector["grounding_acc"]
    else:
        vector["vqa_accuracy"] = compute_vqa(pred, gold)
        composite = vector["vqa_accuracy"]
    vector["footage_composite"] = composite
    return vector


@scorer(metrics=[mean()])
def footage_score() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        output = getattr(state, "output", None)
        text = getattr(output, "completion", "") or ""
        breakdown = compute_footage(text, dict(state.metadata or {}))
        composite = float(breakdown["footage_composite"] or 0.0)
        return Score(
            value=composite,
            answer=str({k: v for k, v in breakdown.items() if v is not None}),
            explanation=f"footage_composite={composite:.2f}",
            metadata=breakdown,
        )

    return score
