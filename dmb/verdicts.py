"""Cross-backend verdict comparison.

Run the same rollout on two backends and compare navigation verdicts.
Fidelity bar (not equality): same found-set, recall within tolerance,
same safety outcome. Used for mock-vs-SITL today and mock-vs-Colosseum
once Track 2 is live (#6).
"""

from __future__ import annotations

from dmb.scorers.mission import compute_mission_scores

RECALL_TOLERANCE = 0.15


def compare_scores(a: dict, b: dict) -> dict:
    """Compare two score vectors; return match verdict + diffs."""
    diffs = {}
    recall_ok = abs(a["rescue_recall"] - b["rescue_recall"]) <= RECALL_TOLERANCE
    if not recall_ok:
        diffs["rescue_recall"] = (a["rescue_recall"], b["rescue_recall"])
    safety_ok = (a["safety"] >= 1.0) == (b["safety"] >= 1.0)
    if not safety_ok:
        diffs["safety"] = (a["safety"], b["safety"])
    found_ok = a["found_any"] == b["found_any"]
    if not found_ok:
        diffs["found_any"] = (a["found_any"], b["found_any"])
    return {"match": not diffs, "diffs": diffs}


def compare_backends(make_a, make_b, metadatas: list[dict], rollout) -> list[dict]:
    """Run `rollout(bridge, metadata)` on both backends per sample; compare."""
    report = []
    for metadata in metadatas:
        states = {}
        for label, make in (("a", make_a), ("b", make_b)):
            bridge = make(dict(metadata))
            try:
                rollout(bridge, dict(metadata))
                states[label] = bridge.get_state()
            finally:
                bridge.close()
        scores = {label: compute_mission_scores(state, 0) for label, state in states.items()}
        verdict = compare_scores(scores["a"], scores["b"])
        report.append(
            {
                "seed": metadata.get("seed"),
                "scores": scores,
                "match": verdict["match"],
                "diffs": verdict["diffs"],
            }
        )
    return report
