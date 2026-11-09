"""Quake/slide harder-eval tasks: same harness as the FIRMS matrix.

Scenarios carry their own victims/hazards/phases; scored by the mission
objective (no spread reference outside FIRMS peat).
"""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def quake():
    return Task(
        dataset=build_samples("earthquake_hard", seeds=[7]),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def quake_fleet():
    return Task(
        dataset=build_samples("earthquake_hard", seeds=[7], fleet_size=2),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def slide():
    return Task(
        dataset=build_samples("landslide_hard", seeds=[7]),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def slide_fleet():
    return Task(
        dataset=build_samples("landslide_hard", seeds=[7], fleet_size=2),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )
