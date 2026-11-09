"""Mission eval: tsunami coastal-surge navigation on the mock backend.

Exercises phased hazards: the surge channel activates at wave-arrival and
the debris field at the debris phase (see scenarios/tsunami.yaml).
"""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.mission import drone_mission


@task
def tsunami():
    return Task(
        dataset=build_samples("tsunami", seeds=[1, 2, 3]),
        solver=[drone_mission()],
        scorer=mission_score(),
    )
