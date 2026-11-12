"""Issue-74 RL gate: deterministic dispatch rewards per checkpoint.

Scores held-out dispatch completions (the 24 T5 rows) with the
verifiable rule reward (``hsea.rlvr.rewards``). Decides the
pre-registered gates: GRPO pilot must beat the RFT baseline; arms
(kto/ipo) compare against the DPO cold-start. No GPU, no API.

Reads ``eval_<ckpt>.jsonl`` (full evals) or
``eval_<ckpt>_dispatch.jsonl`` (dispatch-only gens), whichever exists.

Commands (run from the repo root):
    python scripts/aya74_rl.py dispatch-rewards  # -> dispatch_rewards.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.rlvr.rewards import (  # noqa: E402
    field_match_reward,
    passes_verifier,
    priority_of,
    rule_reward_dispatch,
)

RUN_DIR = ROOT / "runs" / "aya74"
CHECKPOINTS = ("base", "sft", "dpo", "repair", "kto", "ipo", "rft",
               "grpo")


def _load_comps(ckpt: str) -> dict[str, str] | None:
    for name in (f"eval_{ckpt}.jsonl", f"eval_{ckpt}_dispatch.jsonl"):
        path = RUN_DIR / name
        if not path.exists():
            continue
        comps = {}
        for line in open(path):
            rec = json.loads(line)
            comps[rec["key"]] = rec.get("completion", "")
        return comps
    return None


def cmd_dispatch_rewards() -> None:
    prompts = [json.loads(line) for line in
               open(RUN_DIR / "eval_prompts.jsonl")
               if json.loads(line)["track"] == "dispatch"]
    summary: dict = {"n_prompts": len(prompts), "checkpoints": {}}
    for ckpt in CHECKPOINTS:
        comps = _load_comps(ckpt)
        if comps is None:
            print(f"WARNING: no completions for {ckpt}, skipping")
            continue
        rewards, exact, no_footer, pass_v = [], 0, 0, 0
        rows = []
        for p in prompts:
            comp = comps.get(p["key"], "")
            gold = p["meta"]["priority"]
            r = rule_reward_dispatch(comp, gold)
            rewards.append(r)
            if priority_of(comp) == gold:
                exact += 1
            upper = comp.upper()
            if "DECISION:" not in upper and "ACTION:" not in upper:
                no_footer += 1
            if passes_verifier(comp, gold):
                pass_v += 1
            rows.append({"key": p["key"], "gold": gold, "reward": r,
                         "completion": comp})
        n = len(prompts)
        summary["checkpoints"][ckpt] = {
            "n": n, "mean_reward": sum(rewards) / n if n else None,
            "priority_exact": exact / n if n else None,
            "no_footer_rate": no_footer / n if n else None,
            "verifier_pass": pass_v / n if n else None,
        }
        print(f"{ckpt:8} mean={summary['checkpoints'][ckpt]['mean_reward']:.3f} "
              f"exact={exact}/{n} "
              f"nofooter={no_footer}/{n} pass={pass_v}/{n}")
    ckpts = summary["checkpoints"]
    if "grpo" in ckpts and "rft" in ckpts:
        grpo = ckpts["grpo"]["mean_reward"]
        rft = ckpts["rft"]["mean_reward"]
        gate = grpo > rft
        summary["grpo_beats_rft"] = gate
        print(f"GATE grpo({grpo:.3f}) > rft({rft:.3f}): "
              f"{'PASS' if gate else 'FAIL'}")
    with open(RUN_DIR / "dispatch_rewards.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote runs/aya74/dispatch_rewards.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="issue-74 RL gate")
    parser.add_argument("command", choices=["dispatch-rewards"])
    args = parser.parse_args(argv)
    {"dispatch-rewards": cmd_dispatch_rewards}[args.command]()


if __name__ == "__main__":
    main()
