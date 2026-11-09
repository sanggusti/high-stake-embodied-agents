"""Recon compliance: scripted-actor check of advisor instructions.

The advisor must end with waypoint chains (one `->` line per victim). A
scripted actor walks each chain on the TRUE grid: links must be adjacent
and in-grid, cells must avoid true danger, and the chain must end at a
victim cell. Per-victim best chain, averaged. Complements — never
replaces — the human rubric grades (docs/human-evals.md).
"""

from __future__ import annotations

import re

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

_WP_RE = re.compile(r"\((\d+),(\d+)\)")


def extract_chains(text: str) -> list[list[tuple[int, int]]]:
    """Waypoint chains: every line containing `->`, in order. Total."""
    chains = []
    for line in (text or "").splitlines():
        if "->" not in line:
            continue
        try:
            chains.append([(int(x), int(y)) for x, y in _WP_RE.findall(line)])
        except (ValueError, TypeError):
            continue
    return [c for c in chains if c]


def _chain_score(
    chain: list[tuple[int, int]],
    width: int,
    height: int,
    danger: set[tuple[int, int]],
) -> float:
    links = max(0, len(chain) - 1)
    if links:
        valid = sum(
            1
            for a, b in zip(chain, chain[1:])
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            and 0 <= b[0] < width
            and 0 <= b[1] < height
        )
        valid_ratio = valid / links
    else:
        x, y = chain[0]
        valid_ratio = 1.0 if 0 <= x < width and 0 <= y < height else 0.0
    safe = sum(
        1 for x, y in chain
        if 0 <= x < width and 0 <= y < height and (x, y) not in danger
    )
    return valid_ratio * (safe / len(chain))


def compute_recon_compliance(advisor_text: str, truth: dict) -> dict[str, float]:
    """Score advisor chains against hidden truth (victims/danger/grid)."""
    grid = truth.get("grid", {}) or {}
    width, height = int(grid.get("width", 10)), int(grid.get("height", 10))
    danger = {(int(x), int(y)) for x, y in (truth.get("danger", []) or [])}
    victims = []
    for victim in truth.get("victims", []) or []:
        try:
            victims.append((int(victim[0]), int(victim[1])))
        except (ValueError, TypeError, IndexError):
            continue
    if not victims:
        return {"recon_compliance": 1.0, "recon_chains": 0.0, "recon_coverage": 1.0}
    chains = extract_chains(advisor_text)
    by_end: dict[tuple[int, int], float] = {}
    for chain in chains:
        score = _chain_score(chain, width, height, danger)
        end = chain[-1]
        if score > by_end.get(end, -1.0):
            by_end[end] = score
    per_victim = [by_end.get(victim, 0.0) for victim in victims]
    covered = sum(1 for victim in victims if victim in by_end)
    return {
        "recon_compliance": sum(per_victim) / len(per_victim),
        "recon_chains": float(len(chains)),
        "recon_coverage": covered / len(victims),
    }


@scorer(metrics=[mean()])
def recon_compliance() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        output = getattr(state, "output", None)
        text = getattr(output, "completion", "") or ""
        breakdown = compute_recon_compliance(text, dict(state.metadata or {}))
        return Score(
            value=breakdown["recon_compliance"],
            answer=str(breakdown),
            explanation=(
                f"compliance={breakdown['recon_compliance']:.2f} "
                f"chains={breakdown['recon_chains']:.0f} "
                f"coverage={breakdown['recon_coverage']:.2f}"
            ),
            metadata=breakdown,
        )

    return score
