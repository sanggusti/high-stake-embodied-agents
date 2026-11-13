"""HSEA post-training loop diagram for the HF community blog post (Fig 1).

Five stage boxes with straight feed-forward arrows plus a Recover
feedback arc routed fully BELOW the boxes (never through box text),
from Inspect judges back to SafeTune.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a1a2e"
SLATE = "#4a4a68"
ORANGE = "#f59e0b"
RUST = "#b45309"
BOXFACE = "#f4f4f8"

STAGES = [
    ("CuratorKIT", ["flywheel-v2 + safety", "demos + abstention cal", "held-out gates"]),
    ("AlignTune", ["LoRA SFT, 3 epochs", "DPO preference pass", "shared recipe"]),
    ("SafeTune", ["drift audit vs base", "Harden / Steer / Recover", "Gates 1, 2, 3"]),
    ("AgentTune", ["RFT baseline", "GRPO pilot + IPO/KTO", "dispatch rewards"]),
    ("Inspect judges", ["T1 footage to T4 red-team", "T5 dispatch duels", "dual judges + bars"]),
]


def main(out: str) -> None:
    fig = plt.figure(figsize=(19.35, 6.15), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "HSEA post-training loop: curate, align, audit, steer, judge",
            ha="center", va="center", fontsize=22, fontweight="bold", color=INK)

    n = len(STAGES)
    gap = 0.022
    box_w = (1 - 0.04 - gap * (n - 1)) / n
    box_h = 0.52
    y0 = 0.26
    centers = []
    for i, (title, lines) in enumerate(STAGES):
        x0 = 0.02 + i * (box_w + gap)
        box = FancyBboxPatch((x0, y0), box_w, box_h,
                             boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor=BOXFACE, edgecolor=INK, linewidth=2)
        ax.add_patch(box)
        cx = x0 + box_w / 2
        centers.append(cx)
        ax.text(cx, y0 + box_h - 0.07, title, ha="center", va="center",
                fontsize=19, fontweight="bold", color=INK)
        ax.plot([cx - box_w * 0.32, cx + box_w * 0.32],
                [y0 + box_h - 0.12, y0 + box_h - 0.12],
                color=ORANGE, linewidth=4, solid_capstyle="round")
        for j, line in enumerate(lines):
            ax.text(cx, y0 + box_h - 0.24 - j * 0.10, line,
                    ha="center", va="center", fontsize=14, color=SLATE)

    # Straight feed-forward arrows between adjacent boxes (mid height).
    ymid = y0 + box_h / 2
    for i in range(n - 1):
        x0 = 0.02 + i * (box_w + gap)
        tail = (x0 + box_w + 0.002, ymid)
        head = (x0 + box_w + gap - 0.004, ymid)
        ax.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>",
                                     mutation_scale=22, linewidth=2, color=INK,
                                     shrinkA=0, shrinkB=0))

    # Recover arc: bottom-center of Inspect judges to bottom-center of
    # SafeTune, bulging downward into the empty strip below the boxes.
    a = (centers[4], y0)
    b = (centers[2], y0)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22,
                                 linewidth=2.5, color=RUST,
                                 connectionstyle="arc3,rad=-0.30",
                                 shrinkA=1, shrinkB=1))
    ax.text((centers[2] + centers[4]) / 2, 0.045,
            "repair mix re-tune while verdict is Recover",
            ha="center", va="center", fontsize=14, fontstyle="italic", color=RUST)

    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else
         "docs/articles/assets/high-stakes-embodied-agents/pipeline.png")
