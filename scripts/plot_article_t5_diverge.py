"""T5 arm-duel diverging bars for the HF community blog post (Fig 5).

Replaces the grouped arm-wins bars: horizontal diverging rows per arm
per judge, arm wins right vs base wins left, from
runs/aya73-followup/analysis_followup.json. Conventional judge names.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["repair", "grpo", "orpo"]
JUDGES = [
    ("hf-v41", "DeepSeek"),
    ("hf-gemma", "Gemma"),
    ("meta-ember", "Ember"),
]
GREEN = "#15803d"
SLATE = "#475569"


def main(det_path: str, out: str) -> None:
    t5 = json.load(open(det_path))["t5"]
    rows = [(arm, jkey, jlabel) for arm in ARMS for jkey, jlabel in JUDGES]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = list(range(len(rows)))[::-1]
    for yi, (arm, jkey, jlabel) in zip(y, rows):
        cell = t5[arm][jkey]
        aw = cell[arm]
        bw = cell["base"]
        rest = cell.get("tie", 0) + cell.get("inconsistent", 0)
        ax.barh(yi, aw, color=GREEN)
        ax.barh(yi, -bw, color=SLATE)
        ax.text(aw + 0.3, yi, str(aw), va="center", fontsize=10)
        ax.text(-bw - 0.3, yi, str(bw), va="center", ha="right", fontsize=10)
        if rest:
            ax.text(aw + 2.0, yi, f"tie/incon {rest}", va="center", fontsize=8,
                    color="#64748b")
    ax.set_yticks(y, [f"{arm} x {jlabel}" for arm, _, jlabel in rows],
                  fontsize=10)
    ax.set_xlim(-24, 30)
    ax.set_xlabel("duels (of 24, swap-twice): base wins left, arm wins right")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("T5 base-vs-arm duels per judge (Tiny Aya)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=GREEN, label="arm wins"),
                       Patch(color=SLATE, label="base wins")],
              loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "runs/aya73-followup/analysis_followup.json",
         sys.argv[2] if len(sys.argv) > 2 else
         "docs/articles/assets/high-stakes-embodied-agents/issue-73-t5-arms.png")
