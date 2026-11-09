"""Flywheel train eval: teacher data generation on train seeds."""

from inspect_ai import Task, task

from hsea.data.splits import TRAIN_SEEDS
from hsea.scenarios import build_samples
from hsea.scorers.mission import mission_score
from hsea.solvers.mission import drone_mission


@task
def flywheel_train():
    return Task(
        dataset=[
            *build_samples("forest_fire", seeds=TRAIN_SEEDS),
            *build_samples("tsunami", seeds=TRAIN_SEEDS),
        ],
        solver=[drone_mission()],
        scorer=mission_score(),
    )
