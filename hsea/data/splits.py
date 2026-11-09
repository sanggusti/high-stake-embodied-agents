"""Canonical train/test seed splits for the data flywheel.

Train and test seeds are disjoint by construction; export refuses to mix
them. Flywheel evals build samples from these lists so a finetuned model
is always re-evaluated on seeds it never trained on.
"""

from __future__ import annotations

TRAIN_SEEDS: list[int] = list(range(1, 21))
TEST_SEEDS: list[int] = list(range(101, 111))

assert not set(TRAIN_SEEDS) & set(TEST_SEEDS), "splits must be disjoint"


def split_for_seed(seed: int) -> str:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in TEST_SEEDS:
        return "test"
    raise ValueError(f"seed {seed} is in neither TRAIN nor TEST splits")
