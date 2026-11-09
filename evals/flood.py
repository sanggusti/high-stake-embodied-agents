"""Flood harder-eval tasks: same harness as the FIRMS/quake matrices.

Scenarios carry their own victims/hazards/phases; scored by the mission
objective (no spread reference outside FIRMS peat).
"""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def flood():
    return Task(
        dataset=build_samples("flood_hard", seeds=[7]),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def flood_fleet():
    return Task(
        dataset=build_samples("flood_hard", seeds=[7], fleet_size=2),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def flash():
    return Task(
        dataset=build_samples("flash_flood_hard", seeds=[7]),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def flash_fleet():
    return Task(
        dataset=build_samples("flash_flood_hard", seeds=[7], fleet_size=2),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def tsunami():
    return Task(
        dataset=build_samples("tsunami_hard", seeds=[7]),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )


@task
def tsunami_fleet():
    return Task(
        dataset=build_samples("tsunami_hard", seeds=[7], fleet_size=2),
        solver=[drone_mission()],
        scorer=[mission_score()],
    )
