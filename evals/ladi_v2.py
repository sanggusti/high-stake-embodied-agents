"""LADI v2 task (#19): expert multi-labels as yes/no imagery questions.

Corpus: ``MITLL/LADI-v2-dataset`` (CC-BY-4.0) @ pinned revision, test split,
all 12 label questions per image for the first ``HSEA_FOOTAGE_MAX_IMAGES``
images (default 10). Damage-grade labels route to the severity metric.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from hsea.data.footage import FOOTAGE_SYSTEM, build_ladi_samples
from hsea.scorers.footage import footage_score


@task
def ladi_v2():
    return Task(
        dataset=build_ladi_samples(),
        solver=[
            system_message(FOOTAGE_SYSTEM),
            generate(),
        ],
        scorer=footage_score(),
    )
