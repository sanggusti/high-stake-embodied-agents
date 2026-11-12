"""Issue-74 Stage 3 drift audit: #10 harness over base/SFT/DPO.

Builds SafeTune profiles from the recorded Modal generations
(``runs/aya74/eval_{base,sft,dpo}.jsonl``) and runs
``hsea.safety.drift.compare_profiles`` for base-vs-SFT and base-vs-DPO.
No GPU, no API calls: pure local recompute over committed artifacts.

Commands (run from the repo root):
    python scripts/aya74_drift.py drift   # tables + verdicts -> drift.json
    python scripts/aya74_drift.py plot    # docs/reports/issue-74-drift-*.png

The AgentHarm harm-score is a graded rubric (0/0.5/1) over the 8
curator-written agentharm misuse rows; see ``harm_score``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.safety.drift import (  # noqa: E402
    DriftThresholds,
    compare_profiles,
    drift_table_markdown,
)
from hsea.safety.harness import (  # noqa: E402
    agentharm_panel,
    harm_score,
    load_generations,
    overrefusal_panel,
    profile_for_checkpoint,
)

RUN_DIR = ROOT / "runs" / "aya74"
REPORT_DIR = ROOT / "docs" / "reports"
CHECKPOINTS = ("base", "sft", "dpo")
FINETUNES = ("sft", "dpo", "repair", "kto", "ipo", "rft", "grpo")
SEED = 74

# Shared-harness delegates (#10): the implementations live in
# hsea/safety/harness.py; these names stay for backward compatibility
# (tests + reports import them from this module).
__all__ = ["agentharm_panel", "harm_score", "overrefusal_panel",
           "profile_for"]


def _load() -> tuple[list[dict], dict[str, dict[str, str]]]:
    return load_generations(RUN_DIR, ("base",) + FINETUNES)


def profile_for(ckpt: str, prompts: list[dict],
                comps: dict[str, str]) -> dict:
    """SafeTune profile for one checkpoint (safety + red-team severity)."""
    del ckpt  # checkpoint identity plays no role in the profile
    return profile_for_checkpoint(prompts, comps)


def cmd_drift() -> None:
    prompts, completions = _load()
    profiles = {c: profile_for(c, prompts, completions[c])
                for c in completions}
    reports = {}
    for ft in completions:
        if ft == "base":
            continue
        report = compare_profiles(profiles["base"], profiles[ft],
                                  DriftThresholds())
        reports[ft] = report
        print(f"# drift: base vs {ft}")
        print(drift_table_markdown(report), end="")
        print()
    payload = {
        "seed": SEED,
        "source": {"prompts": "runs/aya74/eval_prompts.jsonl",
                   "completions": [f"runs/aya74/eval_{c}.jsonl"
                                   for c in completions]},
        "reports": reports,
        "agentharm": agentharm_panel(prompts, completions),
        "overrefusal": overrefusal_panel(prompts, completions),
    }
    with open(RUN_DIR / "drift.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote runs/aya74/drift.json")


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.load(open(RUN_DIR / "drift.json"))
    over = payload["overrefusal"]
    ckpts = ["base"] + [c for c in payload["reports"] if c != "base"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = ["safety_benign", "agentharm_benign", "abstention_comply"]
    labels = ["safety benign", "agentharm benign", "abstention comply"]
    x = range(len(names))
    width = min(0.22, 0.8 / max(1, len(ckpts)))
    for i, ckpt in enumerate(ckpts):
        vals = [over["checkpoints"][ckpt][n]["rate"] or 0.0
                for n in names]
        pos = [j + (i - (len(ckpts) - 1) / 2) * width for j in x]
        ax.bar(pos, vals, width, label=ckpt)
    ax.set_xticks(list(x), labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("comply / exact rate (higher better)")
    ax.set_title("Over-refusal guard: benign sets per checkpoint (#74)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-74-overrefusal.png", dpi=150)
    plt.close(fig)

    ah = payload["agentharm"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    harm = [ah["checkpoints"][c]["harm_mean"] or 0.0 for c in ckpts]
    ben = [ah["checkpoints"][c]["benign_comply"] or 0.0 for c in ckpts]
    x = range(len(ckpts))
    ax.bar([i - 0.2 for i in x], harm, 0.4, label="misuse harm-score (lower)")
    ax.bar([i + 0.2 for i in x], ben, 0.4, label="benign comply (higher)")
    ax.set_xticks(list(x), ckpts)
    ax.set_ylim(0, 1.08)
    ax.set_title("AgentHarm subset: harm-score vs paired benign (#74)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-74-agentharm.png", dpi=150)
    plt.close(fig)
    print("plots -> docs/reports/issue-74-overrefusal.png "
          "docs/reports/issue-74-agentharm.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="issue-74 drift audit")
    parser.add_argument("command", choices=["drift", "plot"])
    args = parser.parse_args(argv)
    {"drift": cmd_drift, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
