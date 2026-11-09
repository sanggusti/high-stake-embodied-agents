"""Plot flywheel-v2 curation summaries (issue #7 report figure).

Panels: per-source counts bar chart, SFT/DPO pair-length histograms,
RLVR composite histogram (success vs failure negatives).

``.venv/bin/python scripts/plot_flywheel_v2.py --data data/flywheel-v2
--out docs/reports/issue-7-flywheel-v2-curation.png``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flywheel-v2 summaries.")
    parser.add_argument("--data", default="data/flywheel-v2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.data)
    sft = _rows(base / "sft.jsonl")
    dpo = _rows(base / "dpo.jsonl")
    rlvr = _rows(base / "rlvr.jsonl")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("flywheel-v2 curation (issue #7, train split)")

    # Panel 1: per-source counts.
    sft_sources = ["floodnet-vqa", "ladi-v2", "disasterm3", "crisitext", "l2"]
    counts = [sum(1 for r in sft if r["source"] == s) for s in sft_sources]
    counts += [sum(1 for r in dpo if r["provenance"]["source"] == s)
               for s in ("humaid", "crisismmd")]
    counts += [sum(1 for r in rlvr if r["composite"] >= 0.5),
               sum(1 for r in rlvr if r["composite"] < 0.5)]
    labels = ["FN-VQA", "LADI", "DM3", "CrisiT", "L2",
              "HumAID", "CrisisMMD", "RLVR+", "RLVR-"]
    colors = (["#4C78A8"] * 5 + ["#F58518"] * 2 + ["#54A24B", "#E45756"])
    bars = axes[0].bar(labels, counts, color=colors)
    axes[0].set_title("samples per source")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     str(count), ha="center", va="bottom", fontsize=8)

    # Panel 2: pair-length histograms (chars, log x).
    sft_len = np.array([sum(len(t.get("content", "")) for t in r["messages"])
                        for r in sft])
    dpo_len = np.array([len(r["situation"]) + len(r["chosen"]) for r in dpo])
    bins = np.logspace(np.log10(20), np.log10(max(sft_len.max(), 30000)), 40)
    axes[1].hist(sft_len, bins=bins, alpha=0.6, label=f"SFT (n={len(sft)})")
    axes[1].hist(dpo_len, bins=bins, alpha=0.6, label=f"DPO sit+chosen (n={len(dpo)})")
    axes[1].set_xscale("log")
    axes[1].set_title("pair length (chars)")
    axes[1].set_xlabel("chars (log scale)")
    axes[1].legend(fontsize=8)

    # Panel 3: RLVR composite histogram.
    comp = np.array([r["composite"] for r in rlvr])
    axes[2].hist(comp, bins=20, range=(0, 1), color="#79706E")
    axes[2].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_title(f"RLVR composite (n={len(rlvr)})")
    axes[2].set_xlabel("composite (dashed: SFT cutoff)")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
