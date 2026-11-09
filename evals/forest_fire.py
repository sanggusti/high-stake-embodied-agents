"""Sample mission eval: forest-fire drone navigation on the mock backend."""

from inspect_ai import Task, task

from hsea.scenarios import build_samples
from hsea.scorers.mission import mission_score
from hsea.solvers.mission import drone_mission


@task
def forest_fire():
    return Task(
        dataset=build_samples("forest_fire", seeds=[1, 2, 3]),
        solver=[drone_mission()],
        scorer=mission_score(),
    )
