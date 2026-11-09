"""Provider-free baseline tasks: heuristic + random agents on missions."""

from inspect_ai import Task, task

from hsea.scenarios import build_samples
from hsea.scorers.mission import mission_score
from hsea.solvers.baselines import heuristic_mission, random_mission


@task
def forest_fire_heuristic():
    return Task(
        dataset=build_samples("forest_fire", seeds=[1, 2, 3]),
        solver=[heuristic_mission()],
        scorer=mission_score(),
    )


@task
def forest_fire_random():
    return Task(
        dataset=build_samples("forest_fire", seeds=[1, 2, 3]),
        solver=[random_mission()],
        scorer=mission_score(),
    )
