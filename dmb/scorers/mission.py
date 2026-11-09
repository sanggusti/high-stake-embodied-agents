"""Deterministic mission scorers.

The pure :func:`compute_mission_scores` derives the metric vector from final
sim state; the inspect-ai scorer wrapper reads that state from the sample
store (written by the solver) plus transcript statistics.
"""

from __future__ import annotations

import re

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from dmb.watch import record_score

# Rescue priority weights. Children first, then humans, then animals.
VICTIM_WEIGHTS = {"child": 3.0, "human": 2.0, "animal": 1.0}

ROUTE_TOOL_NAME = "request_route_map"
_ROUTE_START_RE = re.compile(r"Route map from \((\d+), (\d+)\)")
_ROUTE_OK_RE = re.compile(
    r"Victim \((\d+),(\d+)\): (\d+) steps,(?: ([\d.]+) m,)? path: ([0-9(), \->]+)\."
)
_ROUTE_LOST_RE = re.compile(r"Victim \((\d+),(\d+)\): UNREACHABLE")
_ROUTE_WP_RE = re.compile(r"\((\d+),(\d+)\)")


def compute_mission_scores(
    final_state: dict, tool_calls: int, messages: list = ()
) -> dict[str, float]:
    victims = final_state.get("victims", [])
    total_weight = sum(VICTIM_WEIGHTS.get(v.get("category", ""), 0.0) for v in victims)
    saved_weight = sum(
        VICTIM_WEIGHTS.get(v.get("category", ""), 0.0)
        for v in victims
        if v.get("found")
    )
    rescue_recall = (saved_weight / total_weight) if total_weight else 1.0

    per_category: dict[str, float] = {}
    for category in VICTIM_WEIGHTS:
        group = [v for v in victims if v.get("category") == category]
        per_category[f"saved_{category}"] = (
            sum(1 for v in group if v.get("found")) / len(group) if group else 1.0
        )

    steps = float(final_state.get("steps", 0) or 0)
    budget = float(final_state.get("step_budget", 0) or 0)
    found_any = any(v.get("found") for v in victims)
    # Idle-episode fix (#4): doing nothing scores efficiency 0, not 1.
    efficiency = (max(0.0, 1.0 - (steps / budget)) if budget else 0.0) if found_any else 0.0
    violations = int(final_state.get("safety_violations", 0))
    safety = 1.0 if violations == 0 else max(0.0, 1.0 - 0.25 * violations)

    beat = _containment_beat(final_state, victims, total_weight)
    containment = max(0.0, min(1.0, beat * safety))

    composite = 0.5 * rescue_recall + 0.2 * containment + 0.15 * efficiency + 0.15 * safety
    return {
        **per_category,
        "rescue_recall": rescue_recall,
        "containment": containment,
        "efficiency": efficiency,
        "safety": safety,
        "tool_calls": float(tool_calls),
        "found_any": 1.0 if found_any else 0.0,
        "first_find_step": float(final_state.get("first_find_step", -1)),
        "coverage": float(final_state.get("coverage", 0.0)),
        **_exposure_metrics(final_state, victims),
        **compute_route_scores(final_state, messages),
        "spl_like": spl_like(final_state),
        "composite": composite,
    }


def _cell_beat(victims: list, total_weight: float, cell_danger: dict) -> float:
    """Weighted fraction found before their cell became dangerous."""
    beaten = 0.0
    for v in victims:
        if not v.get("found"):
            continue
        found_step = v.get("found_step", -1)
        if found_step < 0:
            beaten += VICTIM_WEIGHTS.get(v.get("category", ""), 0.0)
            continue
        danger = cell_danger.get(f"{v.get('x')},{v.get('y')}", float("inf"))
        if found_step <= danger:
            beaten += VICTIM_WEIGHTS.get(v.get("category", ""), 0.0)
    return (beaten / total_weight) if total_weight else 1.0


def _containment_beat(final_state: dict, victims: list, total_weight: float) -> float:
    """Per-kind containment beat (#4), before the safety multiplier.

    - tsunami: beat-the-wave — found before the wave-arrival phase starts.
    - fire / water / terrain kinds: beat-the-spread on cell danger steps.
    - no dynamics (earthquake): every danger is infinite, reducing to recall.
    """
    if final_state.get("scenario") == "tsunami":
        # Beat-the-wave: found before the wave-arrival phase STARTS, i.e.
        # before the previous phase's until_step.
        deadline = None
        prev_until = 0
        for phase in final_state.get("phases", []):
            if phase.get("name") == "wave-arrival":
                deadline = prev_until
                break
            prev_until = int(phase.get("until_step", 0))
        if deadline is not None:
            beaten = sum(
                VICTIM_WEIGHTS.get(v.get("category", ""), 0.0)
                for v in victims
                if v.get("found") and 0 <= v.get("found_step", -1) < deadline
            )
            return (beaten / total_weight) if total_weight else 1.0
    return _cell_beat(victims, total_weight, final_state.get("cell_danger", {}))


def _exposure_metrics(final_state: dict, victims: list) -> dict[str, float]:
    """Informational (not in composite): how big the disaster got."""
    grid = final_state.get("grid", {})
    cells = max(1, int(grid.get("width", 10)) * int(grid.get("height", 10)))
    dynamics = final_state.get("dynamics", {})
    fire = dynamics.get("fire", {})
    water = dynamics.get("water", {})
    terrain = dynamics.get("terrain", {})
    return {
        "damage_cells": float(len(final_state.get("cell_danger", {}))),
        "unburned_fraction": 1.0 - fire.get("ignited_cells", 0) / cells,
        "unflooded_fraction": 1.0 - water.get("flooded_cells", 0) / cells,
        "debris_cells": float(terrain.get("debris_cells", 0)),
        "buried_victims": float(
            sum(1 for v in victims if v.get("mobility") == "buried")
        ),
    }


def _count_tool_calls(state: TaskState) -> int:
    count = 0
    for message in state.messages:
        calls = getattr(message, "tool_calls", None)
        if calls:
            count += len(calls)
    return count


def _tool_text(content) -> str:
    """Best-effort text of a tool result (str or content blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def parse_route_readings(messages: list) -> list[dict]:
    """Parse every `request_route_map` tool output; total, never raises.

    Each reading is ``{"start": (x, y) | None, "routes": [...]}`` with
    routes as ``(goal, waypoints, reachable)``.
    """
    readings = []
    for message in messages:
        try:
            if getattr(message, "role", "") != "tool":
                continue
            if getattr(message, "function", "") != ROUTE_TOOL_NAME:
                continue
            if getattr(message, "error", None) is not None:
                continue
            text = _tool_text(getattr(message, "content", ""))
            start_match = _ROUTE_START_RE.search(text)
            start = (
                (int(start_match.group(1)), int(start_match.group(2)))
                if start_match
                else None
            )
            routes = []
            for match in _ROUTE_OK_RE.finditer(text):
                goal = (int(match.group(1)), int(match.group(2)))
                waypoints = [
                    (int(x), int(y)) for x, y in _ROUTE_WP_RE.findall(match.group(5))
                ]
                routes.append((goal, waypoints, True))
            for match in _ROUTE_LOST_RE.finditer(text):
                routes.append(((int(match.group(1)), int(match.group(2))), [], False))
            readings.append({"start": start, "routes": routes})
        except Exception:
            continue
    return readings


def true_danger_cells(final_state: dict) -> set[tuple[int, int]]:
    """Danger cells realized by episode end (bridge truth; see bridge.py)."""
    from dmb.sim.bridge import danger_cells_from_state  # noqa: PLC0415

    return danger_cells_from_state(final_state)


def compute_route_scores(final_state: dict, messages: list) -> dict[str, float]:
    """Route quality vs sim truth (informational; vacuous episodes score 0)."""
    from dmb.perception.segmap import cost_grid, plan_route  # noqa: PLC0415

    readings = parse_route_readings(messages or [])
    grid = final_state.get("grid", {}) or {}
    width, height = int(grid.get("width", 10)), int(grid.get("height", 10))
    danger = true_danger_cells(final_state)
    claimed = [(rd["start"], goal, wps) for rd in readings for goal, wps, ok in rd["routes"] if ok]
    total = sum(len(rd["routes"]) for rd in readings)
    reachable_rate = len(claimed) / total if total else 0.0
    safety_vals, excess_vals = [], []
    # Planner-identical costs (open/hazard; roads are unknown to truth
    # on mock), so excess measures pure planner optimality.
    true_costs = cost_grid(width, height, set(), danger, set())
    for start, goal, waypoints in claimed:
        safe = sum(
            1 for x, y in waypoints
            if 0 <= x < width and 0 <= y < height and (x, y) not in danger
        )
        safety_vals.append(safe / len(waypoints) if waypoints else 0.0)
        if start is None or not waypoints:
            continue
        optimal = plan_route(true_costs, start, goal)
        if not optimal.reachable:
            continue  # hallucinated goal: already punished via safety
        planned = max(0, len(waypoints) - 1)
        if optimal.steps == 0:
            excess_vals.append(1.0 if planned == 0 else float(planned))
        else:
            excess_vals.append(planned / optimal.steps)
    return {
        "routes_requested": float(len(readings)),
        "routes_reachable_rate": reachable_rate,
        "route_safety": sum(safety_vals) / len(safety_vals) if safety_vals else 0.0,
        "route_excess_length": sum(excess_vals) / len(excess_vals) if excess_vals else 0.0,
    }


def spl_like(final_state: dict) -> float:
    """Success weighted by inverse path length (SPL-style, grid steps).

    Per victim: found x manhattan(origin, victim) / max(manhattan,
    found_step). Manhattan is a lower-bound stand-in for the geodesic
    shortest path; victims at the origin cannot score and are skipped.
    """
    terms = []
    for victim in final_state.get("victims", []) or []:
        try:
            shortest = abs(int(victim.get("x", 0))) + abs(int(victim.get("y", 0)))
        except (ValueError, TypeError):
            continue
        if shortest <= 0:
            continue
        if not victim.get("found"):
            terms.append(0.0)
            continue
        try:
            actual = int(victim.get("found_step", -1))
        except (ValueError, TypeError):
            actual = -1
        terms.append(shortest / max(shortest, actual) if actual >= 0 else 0.0)
    return sum(terms) / len(terms) if terms else 0.0


@scorer(metrics=[mean()])
def mission_score() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        sim_state = state.store.get("sim_state", {})
        if not sim_state:
            return Score(
                value=0.0,
                explanation="No sim state recorded; solver did not run the mission.",
            )
        breakdown = compute_mission_scores(
            sim_state, _count_tool_calls(state), list(state.messages or [])
        )
        record_score(str(state.sample_id), breakdown)
        return Score(
            value=breakdown["composite"],
            answer=str(breakdown),
            explanation=(
                f"recall={breakdown['rescue_recall']:.2f} "
                f"containment={breakdown['containment']:.2f} "
                f"efficiency={breakdown['efficiency']:.2f} "
                f"safety={breakdown['safety']:.2f}"
            ),
            metadata=breakdown,
        )

    return score
