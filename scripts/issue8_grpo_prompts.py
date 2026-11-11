"""Issue #8 GRPO/RFT pilot prompt set builder (stage 5, CPU-only).

Writes ``runs/issue-8/grpo_prompts.jsonl``: 400 dispatch situations
(seed-8 sample of flywheel-v2 dpo.jsonl with gold priority) + 100
safety/abstention prompts (train-slice ITEMS variants + abstention cal
with gold labels). Each row carries the gold dict the rule rewards
(``hsea.rlvr.rewards``) and the RFT verifier-filter consume:

  {"id", "prompt", "gold": {"priority"? , "label"?, "expects_tool_call"}}

The SAME file feeds the GRPO pilot and the E-RFT baseline so the
comparison is apples-to-apples (survey §1.2: RLVR must beat RFT on
the same verifier).

Usage:
  .venv/bin/python scripts/issue8_grpo_prompts.py [--seed 8]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.safety.data import ITEMS, SAFETY_SYSTEM  # noqa: E402

RUN_DIR = ROOT / "runs" / "issue-8"
PROMPTS_PATH = RUN_DIR / "grpo_prompts.jsonl"
DPO_PATH = ROOT / "data" / "flywheel-v2" / "dpo.jsonl"
ABS_SCEN = ROOT / "data" / "abstention" / "scenarios.jsonl"
ABS_BENIGN = ROOT / "data" / "abstention" / "benign.jsonl"
HELDOUT_PATH = RUN_DIR / "safetune_heldout.json"

SEED = 8
N_DISPATCH = 400
N_SAFETY = 100

DISPATCH_SYSTEM = (
    "You are a disaster-response dispatcher. Read the crisis report, "
    "then reply with exactly these lines:\n"
    "DECISION: <refuse|escalate|abstain|comply>\n"
    "CONFIDENCE: <0-100>\n"
    "RESPONSE: <PRIORITY {1|2|3} plus the next action>"
)


def build_prompts(seed: int = SEED) -> list[dict]:
    """Seeded pilot prompt set (dispatch + safety with gold)."""
    rng = random.Random(seed)
    rows: list[dict] = []

    dpo = [json.loads(line) for line in open(DPO_PATH) if line.strip()]
    rng.shuffle(dpo)
    for i, rec in enumerate(dpo[:N_DISPATCH]):
        rows.append({
            "id": f"dispatch-{i:04d}",
            "system": DISPATCH_SYSTEM,
            "prompt": rec["situation"],
            "gold": {"priority": rec["priority"], "label": "comply",
                     "expects_tool_call": False},
        })

    held = set()
    if HELDOUT_PATH.exists():
        for gids in json.load(open(HELDOUT_PATH))["heldout"].values():
            held.update(gids)
    safety: list[dict] = []
    for i, (bench, _cat, label, prompt) in enumerate(ITEMS):
        if f"{bench}:{i}" in held:
            continue
        safety.append({
            "id": f"safety-{bench}-{i}",
            "system": SAFETY_SYSTEM,
            "prompt": prompt,
            "gold": {"label": label,
                     "expects_tool_call": bench == "agentharm"},
        })
    for path in (ABS_SCEN, ABS_BENIGN):
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") != "cal":
                continue
            safety.append({
                "id": f"safety-abs-{rec['id']}",
                "system": SAFETY_SYSTEM,
                "prompt": rec["prompt"],
                "gold": {"label": rec["label"],
                         "expects_tool_call": False},
            })
    rng.shuffle(safety)
    rows.extend(safety[:N_SAFETY])
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_prompts(args.seed)
    with open(PROMPTS_PATH, "w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    print(f"prompts={len(rows)} -> {PROMPTS_PATH}")


if __name__ == "__main__":
    main()
