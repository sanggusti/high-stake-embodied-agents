"""FloodNet-VQA task (#19): native UAV visual QA through a served VLM.

Corpus: ``ljx620/FloodNet-VQA`` @ pinned revision, validation split, first
``HSEA_FOOTAGE_LIMIT`` pairs (default 25) in stream order. Truth answers stay
in metadata for the deterministic footage scorer.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from hsea.data.footage import FOOTAGE_SYSTEM, build_floodnet_samples
from hsea.scorers.footage import footage_score


@task
def floodnet_vqa():
    return Task(
        dataset=build_floodnet_samples(),
        solver=[
            system_message(FOOTAGE_SYSTEM),
            generate(),
        ],
        scorer=footage_score(),
    )
