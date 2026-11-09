"""Quantitative difficulty per sample, computed at generation time.

Deterministic function of (scenario, seed, placements): same inputs always
give the same score. Used for stratification analysis — easy-band samples
should score higher than hard-band ones for any competent agent.
"""

from __future__ import annotations

import math

VISIBILITY_PENALTY = {"low": 1.0, "medium": 0.5, "high": 0.0}


def difficulty_score(metadata: dict) -> dict:
    """Return {'score': 0..1, 'band': easy|medium|hard} for sample metadata."""
    grid = metadata.get("grid", {})
    width, height = int(grid.get("width", 10)), int(grid.get("height", 10))
    placements = metadata.get("victim_placements", [])
    hazards = metadata.get("hazards", [])
    knobs = metadata.get("difficulty", {})

    count_f = min(1.0, len(placements) / 8.0)

    diag = max(1.0, math.hypot(width, height))
    if placements:
        mean_dist = sum(math.hypot(v["x"], v["y"]) for v in placements) / len(placements)
    else:
        mean_dist = 0.0
    dist_f = min(1.0, mean_dist / (diag / 2))

    hazard_cells = 0
    for x in range(width):
        for y in range(height):
            for hz in hazards:
                if math.hypot(x - hz.get("x", -1), y - hz.get("y", -1)) <= float(hz.get("radius", 0)):
                    hazard_cells += 1
                    break
    hazard_f = hazard_cells / max(1, width * height)

    vis_f = VISIBILITY_PENALTY.get(str(knobs.get("visibility", "medium")), 0.5)

    if placements:
        mobile_f = sum(1 for v in placements if v.get("mobility", "static") != "static") / len(placements)
    else:
        mobile_f = 0.0

    score = 0.25 * count_f + 0.30 * dist_f + 0.20 * hazard_f + 0.15 * vis_f + 0.10 * mobile_f
    score = round(min(1.0, max(0.0, score)), 4)
    # Cutoffs calibrated on seeds 1..100 across scenarios (observed 0.40-0.74).
    band = "easy" if score < 0.55 else ("medium" if score < 0.65 else "hard")
    return {"score": score, "band": band}
