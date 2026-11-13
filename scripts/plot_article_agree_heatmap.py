"""Three-judge agreement heatmap for the HF community blog post (Fig 11).

Replaces the grouped bars: tracks as rows, judge pairs as columns,
cells annotated with agree/n plus kappa, from
runs/aya74/agreement3.json. Judge pairs use conventional model names.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS = [
    ("hf-v41-vs-hf-gemma", "DeepSeek x Gemma"),
    ("hf-v41-vs-meta-ember", "DeepSeek x Ember"),
    ("hf-gemma-vs-meta-ember", "Gemma x Ember"),
]
TRACKS = [
    ("T2", "T2 safety"),
    ("T3", "T3 abstention"),
    ("T4", "T4 red-team"),
    ("T1", "T1 footage"),
]


def main(det_path: str, out: str) -> None:
    d = json.load(open(det_path))["single"]
    agree = np.zeros((len(TRACKS), len(PAIRS)))
    annot = []
    for i, (tkey, _) in enumerate(TRACKS):
        row = []
        for j, (pkey, _) in enumerate(PAIRS):
            c = d[tkey]["pairs"][pkey]
            agree[i, j] = c["agree"] / c["n"]
            row.append(f"{c['agree']}/{c['n']}\nk {c['kappa']:.2f}")
        annot.append(row)
    ns = [d[t]["n"] for t, _ in TRACKS]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    im = ax.imshow(agree, vmin=0.5, vmax=1.0, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(PAIRS)), [label for _, label in PAIRS], fontsize=11)
    ax.set_yticks(range(len(TRACKS)),
                  [f"{label} (n={n})" for (_, label), n in zip(TRACKS, ns)],
                  fontsize=11)
    for i in range(len(TRACKS)):
        for j in range(len(PAIRS)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("verdict agreement", fontsize=11)
    ax.set_title("Three-judge agreement per track (Aya Expanse, dual pair + third judge)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "runs/aya74/agreement3.json",
         sys.argv[2] if len(sys.argv) > 2 else
         "docs/articles/assets/high-stakes-embodied-agents/issue-74-agreement3.png")
