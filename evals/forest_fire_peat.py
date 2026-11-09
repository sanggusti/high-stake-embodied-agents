"""Peatland-fire mission eval: smolder-to-breakout dynamics on the mock backend."""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def forest_fire_peat():
    return Task(
        dataset=build_samples("forest_fire_peat", seeds=[1, 2, 3]),
        solver=[drone_mission()],
        scorer=mission_score(),
    )
