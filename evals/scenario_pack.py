"""Scenario-pack sweep: base disasters x fixed seeds (baseline table).

Large variants (scenarios with a ``variant_of`` key) are excluded: they run
via their own tasks so the published pack table stays comparable.
"""

from inspect_ai import Task, task

from hsea.scenarios import build_samples, pack_scenarios
from hsea.scorers.mission import mission_score
from hsea.solvers.mission import drone_mission

PACK_SEEDS = [1, 2, 3]


@task
def scenario_pack():
    dataset = []
    for scenario in pack_scenarios():
        dataset.extend(build_samples(scenario, seeds=PACK_SEEDS))
    return Task(
        dataset=dataset,
        solver=[drone_mission()],
        scorer=mission_score(),
    )
