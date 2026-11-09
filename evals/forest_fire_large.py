"""Large-area forest-fire mission eval (20x20 variant)."""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def forest_fire_large():
    return Task(
        dataset=build_samples("forest_fire_large", seeds=[1, 2, 3]),
        solver=[drone_mission()],
        scorer=mission_score(),
    )
