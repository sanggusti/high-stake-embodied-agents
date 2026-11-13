"""Tiny Aya before/after slope chart for the HF community blog post (Fig 2).

Replaces the grouped bars: one slope per metric across base, SFT, and
DPO finetune, from runs/aya73-recompute/before_after.json. Monotonic
climbs read as rising slopes; the SFT-peak-then-retreat shape on
abstention and red-team ASR reads as a kink.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = {
    "safety label-match (full 78)": "safety",
    "refusal gate: harmbench+airbench": "refusal gate",
    "benign comply": "benign comply",
    "abstention soft accuracy": "abstention soft",
    "abstention eval-split soft accuracy": "abstention split",
    "redteam ASR (lower better)": "red-team ASR (lower better)",
    "footage probe composite (text-only, 60)": "footage",
}
BLUE = "#2563eb"
RUST = "#b45309"


def main(det_path: str, out: str) -> None:
    rows = json.load(open(det_path))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    xs = [0, 1, 2]
    for m in rows:
        name = m["metric"]
        ys = [m["base"]["rate"], m["sft"]["rate"], m["finetune"]["rate"]]
        bad = "lower better" in name
        color = RUST if bad else BLUE
        ax.plot(xs, ys, marker="o", color=color, linewidth=2.5,
                alpha=0.9 if not bad else 1.0)
        label = SHORT.get(name, name)
        dy = {"safety": 0.022, "refusal gate": -0.024,
              "abstention soft": 0.018, "abstention split": -0.02}.get(label, 0.0)
        ax.text(2.04, ys[2] + dy, f"{label} {ys[2]:.3f}", va="center",
                fontsize=10, color=color)
        ax.text(-0.04, ys[0], f"{ys[0]:.3f}", va="center", ha="right",
                fontsize=9, color=color)
    ax.set_xticks(xs, ["base", "SFT", "DPO finetune"], fontsize=12)
    ax.set_xlim(-0.25, 2.75)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("rate")
    ax.set_title("Tiny Aya: SFT does the work, DPO extends safety and pays abstention + red-team")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "runs/aya73-recompute/before_after.json",
         sys.argv[2] if len(sys.argv) > 2 else
         "docs/articles/assets/high-stakes-embodied-agents/issue-73-delta-bars.png")
