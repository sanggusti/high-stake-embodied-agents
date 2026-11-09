"""Conditions probe: same world under different weather/time-of-day."""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def conditions():
    return Task(
        dataset=[
            *build_samples("forest_fire", seeds=[1]),
            *build_samples("forest_fire", seeds=[1], weather="clear", time_of_day="day"),
            *build_samples("forest_fire", seeds=[1], weather="haze-heavy", time_of_day="night"),
        ],
        solver=[drone_mission()],
        scorer=mission_score(),
    )
