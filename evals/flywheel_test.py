"""Flywheel test eval: held-out seeds for base-vs-finetuned delta."""

from inspect_ai import Task, task

from hsea.data.splits import TEST_SEEDS
from hsea.scenarios import build_samples
from hsea.scorers.mission import mission_score
from hsea.solvers.mission import drone_mission


@task
def flywheel_test():
    return Task(
        dataset=[
            *build_samples("forest_fire", seeds=TEST_SEEDS),
            *build_samples("tsunami", seeds=TEST_SEEDS),
        ],
        solver=[drone_mission()],
        scorer=mission_score(),
    )
