"""Live SAM route probe: real API segmentation on mock renders.

Scripted solver (model id unused): one survey episode through
`SAMEdgePerception` with `sam_from_env()`. Costs ~$0.05/sample and is
EXPECTED to return empty seg-maps on mock renders (honest degradation);
it proves the live wiring, scan budget, and stale reuse end to end.
For route quality, use native backends (see sam-segmap.md runbook).

Requires META_AI_API_KEY (or DMB_SAM_API_KEY) exported in the environment.
"""

from inspect_ai import Task, task

from dmb.scenarios import build_samples
from dmb.scorers.mission import mission_score
from dmb.solvers.baselines import live_sam_route_mission


@task
def routes_sam_live():
    return Task(
        dataset=build_samples("flood", seeds=[1]),
        solver=[live_sam_route_mission()],
        scorer=mission_score(),
    )
