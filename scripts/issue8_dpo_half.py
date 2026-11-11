"""Issue #8 DPO-half contingency splitter (CPU-only, seed 8).

Contingency for the DPO cost gate: if the full 7,000-pair DPO prices
beyond the tail budget, train on a stratified half (3,499 pairs,
priority-proportional, seed 8) instead. Full-data DPO is the plan;
this file exists so the fallback is mechanical, not improvised, and
lands in the DEVIATIONS table either way.

Usage:
  .venv/bin/python scripts/issue8_dpo_half.py   # writes runs/issue-8/dpo_half.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 8
DPO_PATH = ROOT / "data" / "flywheel-v2" / "dpo.jsonl"
HALF_PATH = ROOT / "runs" / "issue-8" / "dpo_half.jsonl"


def build_half(seed: int = SEED) -> list[dict]:
    """Stratified half (priority-proportional, seeded)."""
    rng = random.Random(seed)
    rows = [json.loads(line) for line in open(DPO_PATH) if line.strip()]
    by_pri: dict[str, list[dict]] = {}
    for rec in rows:
        by_pri.setdefault(rec["priority"], []).append(rec)
    half: list[dict] = []
    for pri in sorted(by_pri):
        pool = by_pri[pri]
        rng.shuffle(pool)
        half.extend(pool[: len(pool) // 2])
    rng.shuffle(half)
    return half


def main() -> None:
    half = build_half()
    HALF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HALF_PATH, "w") as fh:
        for rec in half:
            fh.write(json.dumps(rec) + "\n")
    print(f"half={len(half)} -> {HALF_PATH}")


if __name__ == "__main__":
    main()
