"""Safety stack task (#14): all four benches + custom abstention set.

Corpus: built-in curator-written subsets in ``hsea.safety.data`` (see
provenance notes there). ``HSEA_SAFETY_BENCH`` selects one bench or
``all`` (default); ``HSEA_SAFETY_LIMIT`` caps items per bench (0 = all).
"""

import os

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message

from hsea.safety.data import BENCHES, ITEMS, SAFETY_SYSTEM
from hsea.scorers.safety import safety_score


def build_safety_samples(bench: str = "all",
                         limit: int = 0) -> list[Sample]:
    """Samples for one bench or the full stack, in dataset order."""
    if bench != "all" and bench not in BENCHES:
        raise ValueError(f"unknown bench {bench!r}; want one of {BENCHES}")
    counts: dict[str, int] = {}
    samples = []
    for b, category, label, prompt in ITEMS:
        if bench != "all" and b != bench:
            continue
        counts[b] = counts.get(b, 0) + 1
        if limit > 0 and counts[b] > limit:
            continue
        samples.append(Sample(
            input=prompt,
            target=label,
            metadata={"bench": b, "category": category, "label": label},
        ))
    return samples


@task
def safety():
    bench = os.environ.get("HSEA_SAFETY_BENCH", "all")
    try:
        limit = int(os.environ.get("HSEA_SAFETY_LIMIT", "0"))
    except ValueError:
        limit = 0
    return Task(
        dataset=build_safety_samples(bench=bench, limit=limit),
        solver=[system_message(SAFETY_SYSTEM), generate()],
        scorer=safety_score(),
    )
