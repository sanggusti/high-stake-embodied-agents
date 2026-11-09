"""License clearance gate for flywheel-v2 curation (issue #7, terms from #24).

Every source the v2 exporter touches must be ``cleared`` here; anything
``skipped`` / ``blocked`` / ``eval-only`` aborts the export. Terms were
resolved in ``docs/license-clearance-24.md``; rows below cite the ruling.
"""

from __future__ import annotations

# source -> (disposition, license/terms, ruling note).
# Dispositions: cleared | skipped | blocked | eval-only.
CLEARANCE: dict[str, tuple[str, str, str]] = {
    "floodnet-vqa": (
        "cleared",
        "CDLA-Permissive-1.0 (base FloodNet; ljx620 mirror card untagged)",
        "QA text + image refs only, no pixels; gated research-only (#7).",
    ),
    "ladi-v2": (
        "cleared",
        "CC-BY-4.0",
        "Official MITLL upload (#24 §1 row 7 covers LADI terms).",
    ),
    "disasterm3": (
        "cleared",
        "CC-BY-NC-SA-4.0",
        "NC/SA research-only, non-commercial, share-alike (#24 §2).",
    ),
    "crisitext": (
        "cleared",
        "CC-BY-4.0",
        "Open; FEMA/GTD situations only, synthetic rows filtered.",
    ),
    "humaid": (
        "cleared",
        "CC-BY-NC-SA-4.0",
        "NC research-only; labels-only, no tweet text (Twitter ToS).",
    ),
    "crisismmd": (
        "cleared",
        "CC-BY-NC-SA-4.0",
        "NC research-only; labels-only, no tweet text/images.",
    ),
    "l2": (
        "cleared",
        "own bench data",
        "inspect-ai logs + runs/ trajectories generated in-repo.",
    ),
    "trec-is": (
        "skipped",
        "TREC/NIST research + X ToS, no redistribution",
        "SKIPPED per user decision 2026-11-09 (X key + ToS, #24 §1 row 10).",
    ),
    "vidi": (
        "blocked",
        "none stated",
        "BLOCKED #24 §1 row 1: no license + YouTube ToS, no grant.",
    ),
    "sen1floods11": (
        "blocked",
        "none stated",
        "BLOCKED #24 §1 row 2 until written terms; SKIPPED (author email).",
    ),
    "crisishelpoffer": (
        "blocked",
        "none (unreleased, on-request)",
        "BLOCKED #24 §1 row 11 until authors release; SKIPPED.",
    ),
    "rescuenet": (
        "eval-only",
        "CC BY-NC-ND 4.0",
        "ND bars training derivatives; held-out eval only (#24 §2).",
    ),
    "sard": (
        "skipped",
        "CC BY 4.0 but paid DataPort Standard access",
        "SKIPPED per user decision 2026-11-09 (#24 §1 row 5).",
    ),
    "flame-3": (
        "skipped",
        "CC BY 4.0 but IEEE signup",
        "SKIPPED per user decision 2026-11-09 (#24 §1 row 8).",
    ),
    "heridal": (
        "skipped",
        "CC Attribution 3.0, confirm outstanding",
        "SKIPPED per user decision 2026-11-09 (#24 §1 row 6).",
    ),
    "dad": (
        "skipped",
        "CrisisNLP ToS, research-only",
        "SKIPPED per user decision 2026-11-09; use MEDIC bundle (#24 §1).",
    ),
}

CLEARED_SOURCES = tuple(s for s, (d, _, _) in CLEARANCE.items() if d == "cleared")


def disposition(source: str) -> tuple[str, str, str]:
    """Return (disposition, license, note) for a source id.

    Unknown ids are refused: fail closed, never pass through.
    """
    try:
        return CLEARANCE[source]
    except KeyError:
        raise ValueError(
            f"unknown source {source!r}: not in the #24 clearance table, refused"
        ) from None


def assert_cleared_for_training(sources: list[str] | tuple[str, ...]) -> None:
    """Abort unless every source is cleared for v2 training. No-op otherwise."""
    refused = []
    for source in sources:
        status, _terms, note = disposition(source)
        if status != "cleared":
            refused.append(f"{source} ({status}: {note})")
    if refused:
        raise ValueError(
            "refusing v2 export with non-cleared sources: " + "; ".join(refused)
        )
