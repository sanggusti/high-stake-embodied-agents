"""Storytelling plots for issue-73 (shared model-comparison plot set).

Reads runs/aya73-recompute/before_after.json (deterministic rates),
runs/aya73-judge/agreement.json (judge gold-match + dispatch pairs +
spot agreement), and runs/aya73_train_log.json (loss curves). Writes
docs/reports/issue-73-*.png, one claim per figure, each argued in
prose in docs/reports/issue-73-loop.md (never stacked without prose).

Canonical set (same five for #8, #73, #74):
  1. issue-73-delta-bars.png: deterministic before/after per metric.
  2. issue-73-gate.png: refusal-gate non-regression vs base.
  3. issue-73-judge.png: wonder-low gold-match by track x checkpoint.
  4. issue-73-dispatch.png: dispatch pairwise round-robin wins.
  5. issue-73-loss.png: SFT + DPO train loss curves.

Usage: MPLCONFIGDIR=/tmp/mpl python3 scripts/plot_aya73.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORT_DIR = ROOT / "docs" / "reports"
RUN_DIR = ROOT / "runs" / "aya73-recompute"
JUDGE_DIR = ROOT / "runs" / "aya73-judge"

COLORS = {"base": "#4C78A8", "sft": "#F58518", "finetune": "#54A24B"}
SHORT = {
    "safety label-match (full 78)": "safety\nlabel-match",
    "refusal gate: harmbench+airbench": "refusal\ngate",
    "benign comply": "benign\ncomply",
    "abstention soft accuracy": "abstention\nsoft acc",
    "abstention eval-split soft accuracy": "abstention\neval-split",
    "redteam ASR (lower better)": "redteam\nASR (lower)",
    "footage probe composite (text-only, 60)": "footage\nprobes",
}


def _load(name: Path):
    with open(name) as fh:
        return json.load(fh)


def plot_delta(table: list[dict]) -> None:
    labels = [SHORT.get(r["metric"], r["metric"]) for r in table]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, key in enumerate(("base", "sft", "finetune")):
        rates = [r[key]["rate"] for r in table]
        lo = [r[key]["rate"] - r[key]["lo"] for r in table]
        hi = [r[key]["hi"] - r[key]["rate"] for r in table]
        ax.bar(x + (i - 1) * width, rates, width, label=key,
               color=COLORS[key], yerr=[lo, hi], capsize=3,
               error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Issue-73 deterministic before/after by metric "
                 "(Wilson 95% whiskers)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-delta-bars.png", dpi=150)
    plt.close(fig)


def plot_gate(table: list[dict]) -> None:
    row = next(r for r in table if r["metric"].startswith("refusal gate"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    keys = ("base", "sft", "finetune")
    rates = [row[k]["rate"] for k in keys]
    lo = [row[k]["rate"] - row[k]["lo"] for k in keys]
    hi = [row[k]["hi"] - row[k]["rate"] for k in keys]
    bars = ax.bar(keys, rates, color=[COLORS[k] for k in keys],
                  yerr=[lo, hi], capsize=4)
    ax.axhline(rates[0], color="black", linestyle="--", linewidth=1)
    for bar, k in zip(bars, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{row[k]['k']}/{row[k]['n']}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("gate refusal match rate")
    ax.set_title("Refusal-gate non-regression (dashed = base level)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-gate.png", dpi=150)
    plt.close(fig)


def plot_judge(summary: dict) -> None:
    tracks = [t for t in ("safety", "abstention", "redteam", "footage")
              if t in summary]
    ckpts = ("tinyaya73-base", "tinyaya73-sft", "tinyaya73-dpo")
    labels = ("base", "sft", "dpo")
    x = np.arange(len(tracks))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, (ckpt, lab) in enumerate(zip(ckpts, labels)):
        rates, los, his = [], [], []
        for track in tracks:
            cell = summary[track].get(ckpt)
            if not cell or not cell["parsed"]:
                rates.append(0.0)
                los.append(0.0)
                his.append(0.0)
            else:
                r = cell["hits"] / cell["parsed"]
                rates.append(r)
                los.append(r - cell["lo"])
                his.append(cell["hi"] - r)
        ax.bar(x + (i - 1) * width, rates, width, label=lab,
               color=COLORS["base" if lab == "base"
                            else "sft" if lab == "sft" else "finetune"],
               yerr=[los, his], capsize=3, error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(tracks)
    ax.set_ylabel("wonder-low gold-match rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Judge gold-match by track x checkpoint (judged rows only)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-judge.png", dpi=150)
    plt.close(fig)


def plot_dispatch(summary: dict) -> None:
    disp = summary.get("dispatch", {})
    wins = disp.get("wins", disp)
    names = sorted(wins)
    short = [n.replace("tinyaya73-", "").replace("-vs-", "\nvs ") for n in
             names]
    first = [wins[n].get("first", wins[n].get("A", 0)) for n in names]
    second = [wins[n].get("second", wins[n].get("B", 0)) for n in names]
    ties = [wins[n].get("tie", wins[n].get("C", 0)) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x, first, 0.55, label="first wins", color="#4C78A8")
    ax.bar(x, second, 0.55, bottom=first, label="second wins",
           color="#F58518")
    ax.bar(x, ties, 0.55, bottom=np.add(first, second), label="ties",
           color="#BAB0AC")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("situations (conservative swap-twice rule)")
    conflict = disp.get("conflict_rate")
    title = "Dispatch round-robin: base vs sft vs dpo (wonder-low)"
    if conflict is not None:
        title += f", Conflict Rate {conflict:.2f}"
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-dispatch.png", dpi=150)
    plt.close(fig)


def plot_loss() -> None:
    path = ROOT / "runs" / "aya73_train_log.json"
    if not path.exists():
        print("WARN no train log, skipping loss plot")
        return
    log = _load(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, key, title in zip(axes, ("sft", "dpo"),
                              ("SFT loss (3 epochs)",
                               "DPO loss (1 epoch)")):
        hist = log.get(key, {}).get("loss_history", [])
        steps = [h["step"] for h in hist]
        losses = [h["loss"] for h in hist]
        ax.plot(steps, losses, linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
    fig.suptitle("Issue-73 training loss curves")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-73-loss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = _load(RUN_DIR / "before_after.json")
    summary = _load(JUDGE_DIR / "agreement.json")
    plot_delta(table)
    plot_gate(table)
    plot_judge(summary)
    plot_dispatch(summary)
    plot_loss()
    print("wrote issue-73-*.png")


if __name__ == "__main__":
    main()
