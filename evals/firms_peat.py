"""FIRMS sim-to-real mission eval: real Kalimantan peat ignitions.

Same mission harness as forest_fire_peat, but scenarios carry real VIIRS
ignitions + day-1..7 spread references, scored by both the mission
objective and the FIRMS spread-overlap metric.
"""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.firms_spread import firms_spread_score
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def firms_peat():
    return Task(
        dataset=build_samples("firms_peat_kalimantan_01", seeds=[7]),
        solver=[drone_mission()],
        scorer=[mission_score(), firms_spread_score()],
    )


@task
def firms_peat_fleet():
    return Task(
        dataset=build_samples(
            "firms_peat_kalimantan_01", seeds=[7], fleet_size=2
        ),
        solver=[drone_mission()],
        scorer=[mission_score(), firms_spread_score()],
    )
