"""Recon-advisor task: LM directs rescuers from a real drone survey.

Each sample embeds an oracle route-map text (drone survey) plus staging
info; true victims/danger stay in metadata for the compliance scorer.
The advisor must end with one waypoint line per victim (`path: ...`)."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message

from dmb.scenarios import build_samples, load_scenario
from dmb.scorers.recon import recon_compliance
from dmb.sim.bridge import danger_cells_from_state
from dmb.solvers.baselines import run_oracle_route

RECON_VIGNETTES = (("flood", 1), ("forest_fire", 2), ("flood", 3))

ADVISOR_SYSTEM = (
    "You are a reconnaissance advisor. Give short, ordered, actionable "
    "instructions a stressed rescue team can follow. End with one line "
    "per victim, exactly: path: (x,y) -> (x,y) -> ..."
)


def build_recon_dataset() -> list[Sample]:
    """Deterministic vignettes: oracle survey input + hidden truth metadata."""
    samples = []
    for scenario, seed in RECON_VIGNETTES:
        [base] = build_samples(scenario, [seed])
        metadata = dict(base.metadata)
        metadata["sample_id"] = f"recon-{scenario}-seed-{seed}"
        final, route_text = run_oracle_route(metadata)
        victims = [(v["x"], v["y"], v["category"]) for v in final["victims"]]
        grid = dict(final["grid"])
        width, height = int(grid["width"]), int(grid["height"])
        start = (int(final["drone"]["x"]), int(final["drone"]["y"]))
        kind = load_scenario(scenario).kind
        samples.append(
            Sample(
                input=(
                    f"{kind} disaster, sector {scenario}. A drone survey "
                    f"(assumed geometry) reports:\n{route_text}\n"
                    f"Rescue team staged at {start} on a {width}x{height} "
                    f"grid (x=0..{width - 1}, y=0..{height - 1}). Direct "
                    f"the team: ordered instructions to reach each "
                    f"reported victim, and what to avoid."
                ),
                id=f"recon-{scenario}-seed-{seed}",
                metadata={
                    "scenario": scenario,
                    "seed": seed,
                    "world_hash": metadata["world_hash"],
                    "grid": grid,
                    "victims": victims,
                    "danger": sorted(danger_cells_from_state(final)),
                    "start": start,
                },
            )
        )
    return samples


@task
def recon_advisor():
    return Task(
        dataset=build_recon_dataset(),
        solver=[
            system_message(ADVISOR_SYSTEM),
            generate(),
        ],
        scorer=recon_compliance(),
    )
