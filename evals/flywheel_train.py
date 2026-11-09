"""Flywheel train eval: teacher data generation on train seeds."""

from inspect_ai import Task, task

from dmb.data.splits import TRAIN_SEEDS
from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


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
