"""Red-team attack-set loader (#22).

``data/redteam/attacks.jsonl`` holds one JSON object per line: id, family
(``jailbreak`` | ``retrieval_injection`` | ``misuse``), variant, severity
(``high`` | ``medium`` | ``low``), prompt, request (short label), and
safetune_probe (SafeTune repair deep-link for #10).
"""

from __future__ import annotations

import json
from pathlib import Path

FAMILIES = ("jailbreak", "retrieval_injection", "misuse")
SEVERITIES = ("high", "medium", "low")

ATTACKS_PATH = Path(__file__).resolve().parents[2] / "data" / "redteam" / "attacks.jsonl"

REQUIRED_KEYS = (
    "id",
    "family",
    "variant",
    "severity",
    "prompt",
    "request",
    "safetune_probe",
)


def load_attacks(path: str | Path = ATTACKS_PATH) -> list[dict]:
    """Load + validate the attack set (stable file order)."""
    records = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            raise ValueError(f"{path}:{lineno}: missing keys {missing}")
        if rec["family"] not in FAMILIES:
            raise ValueError(f"{path}:{lineno}: bad family {rec['family']!r}")
        if rec["severity"] not in SEVERITIES:
            raise ValueError(f"{path}:{lineno}: bad severity {rec['severity']!r}")
        markers = rec.get("markers", [])
        if rec["family"] == "retrieval_injection" and not markers:
            raise ValueError(f"{path}:{lineno}: injection sample needs markers")
        if not isinstance(markers, list):
            raise ValueError(f"{path}:{lineno}: markers must be a list")
        records.append(rec)
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate attack ids")
    return records
