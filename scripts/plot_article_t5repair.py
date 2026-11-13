"""T5 repair-duel bars with conventional judge names (Fig 10).

Same grouped bars as scripts/aya74_eval.py from
runs/aya74/analysis_t5_repair.json (+ agreement3_t5_repair.json Meta
top-up), with the internal hf-v41 / hf-gemma / meta-ember legend
replaced by conventional model names.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "hf-v41": "DeepSeek-V4.1-Flash",
    "hf-gemma": "Gemma-3-27B",
    "meta-ember": "Meta Ember-1.0",
}


def main(run_dir: str, out: str) -> None:
    run = Path(run_dir)
    t5r = json.load(open(run / "analysis_t5_repair.json"))["t5"]
    t5m_path = run / "agreement3_t5_repair.json"
    t5m = json.load(open(t5m_path))["t5"] if t5m_path.exists() else {}
    judges = list(t5r) + [j for j in t5m if j not in t5r]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = ["repair wins", "base wins", "tie", "inconsistent"]
    keys = ["repair", "base", "tie", "inconsistent"]
    x = range(len(labels))
    width = min(0.22, 0.8 / max(1, len(judges)))
    for i, judge in enumerate(judges):
        cell = t5r.get(judge, t5m.get(judge, {}))
        vals = [cell.get(k, 0) for k in keys]
        pos = [j + (i - (len(judges) - 1) / 2) * width for j in x]
        ax.bar(pos, vals, width, label=NAMES.get(judge, judge))
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("dispatch pairs (n=24, swap-twice)")
    ax.set_title("T5 dispatch duels: repair vs base per judge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "runs/aya74",
         sys.argv[2] if len(sys.argv) > 2 else
         "docs/articles/assets/high-stakes-embodied-agents/issue-74-t5-repair.png")
