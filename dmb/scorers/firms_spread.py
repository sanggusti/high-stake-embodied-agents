"""FIRMS sim-to-real spread-overlap scorer.

Compares the simulated burned perimeter against real VIIRS day-1..7
hotspot cells carried in the sample ``firms`` metadata (see the
``firms`` block in ``scenarios/firms_peat_*.yaml``).

The pure :func:`compute_spread_scores` derives the metric vector from
final sim state; the inspect-ai scorer wrapper reads that state from
the sample store (written by the solver) plus the firms reference.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from dmb.watch import record_score

Cell = tuple[int, int]
TOL_CELLS = 15  # 75 m at 5 m/cell: one sub-pixel VIIRS tolerance


def _as_cells(cells: Iterable[Sequence[int]]) -> set[Cell]:
    return {(int(c[0]), int(c[1])) for c in cells}


def _within_tol(cell: Cell, others: set[Cell], tol: int) -> bool:
    return any(math.dist(cell, other) <= tol + 1e-9 for other in others)


def sim_burned_cells(sim_state: dict) -> set[Cell]:
    """Burned/dangerous cells from final sim state (flat cell_danger)."""
    cells: set[Cell] = set()
    for key in (sim_state or {}).get("cell_danger", {}):
        x, y = key.split(",")
        cells.add((int(x), int(y)))
    return cells


def compute_spread_scores(
    sim_state: dict, firms: dict, tol: int = TOL_CELLS
) -> dict[str, float]:
    """Pure overlap of simulated burned cells vs real new-hotspot cells.

    ``spread_recall_75m``: fraction of real day-1..7 cells within ``tol``
    of any simulated burned cell (did the sim reach where fire went?).
    ``spread_precision_75m``: fraction of simulated burned cells within
    ``tol`` of any real (ignition or new) cell (did the sim burn only
    where fire was?). ``spread_extent_ratio``: sim burned count over
    real total (ignitions + new cells).
    """
    sim = sim_burned_cells(sim_state)
    ref = _as_cells((firms or {}).get("ref_new_cells", ()))
    real = ref | _as_cells((firms or {}).get("ign_cells", ()))
    if not sim:
        recall = 0.0
        precision = 0.0
    else:
        recall = (
            sum(_within_tol(c, sim, tol) for c in ref) / len(ref)
            if ref
            else 0.0
        )
        precision = (
            sum(_within_tol(c, real, tol) for c in sim) / len(sim)
            if real
            else 0.0
        )
    extent = len(sim) / len(real) if real else 0.0
    return {
        "spread_recall_75m": recall,
        "spread_precision_75m": precision,
        "spread_extent_ratio": extent,
    }


@scorer(metrics=[mean()])
def firms_spread_score() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        firms = (state.metadata or {}).get("firms")
        if not firms:
            return Score(
                value=0.0,
                explanation="Not a FIRMS sample; no spread reference.",
            )
        sim_state = state.store.get("sim_state", {})
        if not sim_state:
            return Score(
                value=0.0,
                explanation="No sim state recorded; solver did not run.",
            )
        breakdown = compute_spread_scores(sim_state, firms)
        record_score(str(state.sample_id), breakdown)
        return Score(
            value=breakdown["spread_recall_75m"],
            answer=str(breakdown),
            explanation=(
                f"recall={breakdown['spread_recall_75m']:.2f} "
                f"precision={breakdown['spread_precision_75m']:.2f} "
                f"extent={breakdown['spread_extent_ratio']:.2f}"
            ),
            metadata=breakdown,
        )

    return score
