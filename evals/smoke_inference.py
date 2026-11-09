"""Smoke task: proves model routing to any provider with zero tools."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message


@task
def smoke():
    return Task(
        dataset=[
            Sample(
                input="Reply with exactly: SMOKE-OK",
                target="SMOKE-OK",
            )
        ],
        solver=[system_message("You are a smoke test."), generate()],
        scorer=includes(),
    )
