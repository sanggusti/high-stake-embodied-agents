"""Large-area forest-fire mission eval (20x20 variant)."""

from inspect_ai import Task, task

from hsea.scenarios import build_samples
from hsea.scorers.mission import mission_score
from hsea.solvers.mission import drone_mission


@task
def forest_fire_large():
    return Task(
        dataset=build_samples("forest_fire_large", seeds=[1, 2, 3]),
        solver=[drone_mission()],
        scorer=mission_score(),
    )
