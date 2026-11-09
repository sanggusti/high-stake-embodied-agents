"""Pack baselines: scripted policies over the full scenario pack (#7)."""

from inspect_ai import Task, task

from dmb.scenarios import build_samples, pack_scenarios
from dmb.scorers.mission import mission_score
from dmb.solvers.baselines import heuristic_mission, random_mission

PACK_SEEDS = [1, 2, 3]  # same seeds as evals/scenario_pack.py


def _pack():
    dataset = []
    for scenario in pack_scenarios():
        dataset.extend(build_samples(scenario, seeds=PACK_SEEDS))
    return dataset


@task
def pack_heuristic():
    return Task(
        dataset=_pack(),
        solver=[heuristic_mission()],
        scorer=mission_score(),
    )


@task
def pack_random():
    return Task(
        dataset=_pack(),
        solver=[random_mission()],
        scorer=mission_score(),
    )
