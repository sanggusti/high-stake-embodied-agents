"""Oracle route ceiling: perfect perception + real planner on mock.

Scripted solver (no model calls): one high survey scan with oracle
seg-maps, then rescuer routes. Asserts the planner+metrics ceiling —
planner-identical optimal costs, so excess must be 1.0 and safety 1.0
on clean worlds. Run with any model id (unused), e.g. mockllm/model.
"""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.baselines import oracle_route_mission


@task
def routes_oracle():
    return Task(
        dataset=build_samples("flood", seeds=[1, 2, 3]),
        solver=[oracle_route_mission()],
        scorer=mission_score(),
    )
