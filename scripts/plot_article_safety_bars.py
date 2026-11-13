"""Safety-vs-utility grouped bars for the HF community blog post (Fig 8).

Replaces the Safety Pareto scatter (all six checkpoints pile into one
corner) with grouped bars per checkpoint: benign comply rate beside
harmful refuse rate, both from runs/aya74/det_scores.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPTS = ["base", "sft", "dpo", "repair", "kto", "ipo"]
BLUE = "#2563eb"
RUST = "#b45309"


def main(det_path: str, out: str) -> None:
    det = json.load(open(det_path))["checkpoints"]
    comp, ref, comp_n, ref_n = [], [], [], []
    for ck in CKPTS:
        by_label = det[ck]["safety"]["by_label"]
        c, r = by_label["comply"], by_label["refuse"]
        comp.append(c["ok"] / c["n"])
        ref.append(r["ok"] / r["n"])
        comp_n.append((c["ok"], c["n"]))
        ref_n.append((r["ok"], r["n"]))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = range(len(CKPTS))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], comp, w, label="benign comply (utility)",
                color=BLUE)
    b2 = ax.bar([i + w / 2 for i in x], ref, w, label="refuse on harmful (safety)",
                color=RUST)
    for bars, counts in ((b1, comp_n), (b2, ref_n)):
        for bar, (ok, n) in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{ok}/{n}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(x), CKPTS)
    ax.set_ylim(0.72, 1.06)
    ax.set_ylabel("rate")
    ax.set_title("Safety vs utility per checkpoint: benign comply holds while refusal slips")
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    det_default = "runs/aya74/det_scores.json"
    out_default = "docs/articles/assets/high-stakes-embodied-agents/issue-74-safety-pareto.png"
    main(sys.argv[1] if len(sys.argv) > 1 else det_default,
         sys.argv[2] if len(sys.argv) > 2 else out_default)
