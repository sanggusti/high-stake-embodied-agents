"""Render the 3 original Technical-details diagrams for Space issue #59.

Outputs (each asserted <50,000 bytes for the Space asset budget):
  fig-tech-loop.png      Lexsi loop pipeline with safety gates
  fig-tech-dataflow.png  curation -> tune -> eval data flow
  fig-tech-harness.png   eval-harness component map

Palette matches the Space Emberline theme (theme.css). Deterministic:
no random data, fixed layout, DejaVu Sans (matplotlib bundled).

Usage:
  .venv/bin/python scripts/plot_tech_details.py --out-dir /tmp/techfigs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

PAPER = "#F7F5F0"
CARD = "#FFFFFF"
INK = "#14181F"
INK2 = "#1F2733"
EMBER = "#E0531F"
GOLD = "#E9A13B"
TEAL = "#0E9F8A"
MUTED = "#5F6B7A"
LINE = "#E4E1D8"
GATE_BG = "#FDEEE4"
DATA_BG = "#FEF6E8"
OUT_BG = "#E6F5F2"

MAX_BYTES = 50_000


def _box(ax, x, y, w, h, text, sub="", bg=CARD, edge=LINE, tcol=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=bg, edgecolor=edge, linewidth=1.2,
        )
    )
    ax.text(x + w / 2, y + h / 2 + (0.035 if sub else 0), text,
            ha="center", va="center", fontsize=7.5, weight="bold", color=tcol)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.055, sub,
                ha="center", va="center", fontsize=6.2, color=MUTED)


def _diamond(ax, cx, cy, w, h, text, sub=""):
    ax.add_patch(
        Polygon(
            [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
            closed=True, facecolor=GATE_BG, edgecolor=EMBER, linewidth=1.4,
        )
    )
    ax.text(cx, cy + (0.03 if sub else 0), text,
            ha="center", va="center", fontsize=7.5, weight="bold", color=INK)
    if sub:
        ax.text(cx, cy - 0.06, sub,
                ha="center", va="center", fontsize=5.5, color=MUTED)


def _arrow(ax, x1, y1, x2, y2, color=INK2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=1, shrinkB=2),
    )


def _fig(title, subtitle, w=9.0, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.text(5, 2.05, title, ha="center", va="center",
            fontsize=10, weight="bold", color=INK)
    ax.text(5, 1.86, subtitle, ha="center", va="center",
            fontsize=7, color=MUTED)
    return fig, ax


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, facecolor=PAPER, dpi=100)
    plt.close(fig)
    n = path.stat().st_size
    assert n < MAX_BYTES, (name, n)
    print(f"{name}: {n} bytes OK")
    return path


def fig_loop(out_dir: Path) -> Path:
    """Lexsi loop: 5 stages gated 3 times, fail edges return upstream."""
    fig, ax = _fig(
        "HSEA Lexsi loop: stages and safety gates",
        "research-track section 4; gate bars in process.md section 3 and safety-gates.md",
    )
    y, h = 0.55, 0.62
    _box(ax, 0.10, y, 1.18, h, "Curate", "CuratorKIT", bg=DATA_BG, edge=GOLD)
    _box(ax, 1.44, y, 1.18, h, "AlignTune", "SFT + DPO")
    _diamond(ax, 3.25, y + h / 2, 1.05, 0.72, "Gate 1", "refusal")
    _box(ax, 3.95, y, 1.18, h, "SafeTune", "drift repair")
    _diamond(ax, 5.76, y + h / 2, 1.05, 0.72, "Gate 2", "misuse")
    _box(ax, 6.46, y, 1.18, h, "AgentTune", "RLVR + GRPO")
    _diamond(ax, 8.27, y + h / 2, 1.05, 0.72, "Gate 3", "safety")
    _box(ax, 8.97, y, 0.93, h, "Ship", "audit first", bg=OUT_BG, edge=TEAL)
    for x1, x2 in [(1.28, 1.44), (2.62, 2.78), (3.72, 3.95), (5.13, 5.29),
                   (6.23, 6.46), (7.64, 7.80), (8.74, 8.97)]:
        _arrow(ax, x1, y + h / 2, x2, y + h / 2)
    ax.text(5, 0.30, "each gate fail returns to an earlier stage (fail edges in prose, not drawn)",
            ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.text(9.85, 0.10, "HSEA original", ha="right", va="center",
            fontsize=6, color=MUTED)
    return _save(fig, out_dir, "fig-tech-loop.png")


def fig_dataflow(out_dir: Path) -> Path:
    """Corpora -> export splits -> train stages -> eval tables."""
    fig, ax = _fig(
        "Data flow: corpora to splits to checkpoints to eval tables",
        "curate_v2 export; v2 split sizes from issue-7 curation report",
        h=3.8,
    )
    ax.set_ylim(0, 2.4)
    ax.texts[0].set_y(2.12)
    ax.texts[1].set_y(1.94)
    srcs = ["FloodNet-VQA", "LADI v2", "DisasterM3", "HumAID",
            "CrisisMMD", "CrisiText", "bench runs"]
    for i, s in enumerate(srcs):
        yy = 1.52 - i * 0.205
        _box(ax, 0.10, yy, 1.55, 0.17, s, bg=CARD, edge=LINE)
        _arrow(ax, 1.65, yy + 0.085, 2.05, 0.95)
    _box(ax, 2.05, 0.62, 1.85, 0.66, "curate_v2 export", "license gate first",
         bg=DATA_BG, edge=GOLD)
    splits = [("sft.jsonl", "8,673 rows"), ("dpo.jsonl", "7,000 pairs"),
              ("rlvr.jsonl", "250 eps")]
    for i, (a, b) in enumerate(splits):
        yy = 1.52 - i * 0.30
        _box(ax, 4.30, yy, 1.55, 0.24, a, b, bg=DATA_BG, edge=GOLD)
    _arrow(ax, 3.90, 0.95, 4.30, 1.40)
    _arrow(ax, 3.90, 0.95, 4.30, 1.10)
    _arrow(ax, 3.90, 0.95, 4.30, 0.80)
    stages = [("AlignTune", "SFT then DPO"), ("SafeTune", "drift repair"),
              ("AgentTune", "RLVR / GRPO"), ("Inspect evals", "aggregate tables")]
    for i, (a, b) in enumerate(stages):
        xx = 6.25 + i * 0.0
        yy = 1.52 - i * 0.30
        _box(ax, xx, yy, 1.80, 0.24, a, b,
             bg=OUT_BG if i == 3 else CARD, edge=TEAL if i == 3 else LINE)
    _arrow(ax, 5.85, 1.40, 6.25, 1.40)
    _arrow(ax, 5.85, 1.10, 6.25, 1.10)
    _arrow(ax, 5.85, 0.80, 6.25, 0.80)
    for top, bot in ((1.52, 1.46), (1.22, 1.16), (0.92, 0.86)):
        _arrow(ax, 7.15, top, 7.15, bot)
    _box(ax, 8.35, 0.62, 1.55, 0.66, "docs/reports", "tables + plots",
         bg=OUT_BG, edge=TEAL)
    _arrow(ax, 8.05, 0.74, 8.35, 0.95)
    ax.text(9.85, 0.10, "HSEA original", ha="right", va="center",
            fontsize=6, color=MUTED)
    return _save(fig, out_dir, "fig-tech-dataflow.png")


def fig_harness(out_dir: Path) -> Path:
    """Eval tasks -> scorers -> logs -> registry tables and plots."""
    fig, ax = _fig(
        "Eval harness: tasks to scorers to logs to tables and plots",
        "deterministic scorers only; smoke checks routing, never results",
        h=3.8,
    )
    ax.set_ylim(0, 2.4)
    ax.texts[0].set_y(2.12)
    ax.texts[1].set_y(1.94)
    tasks = [("footage tasks", "floodnet_vqa, ladi_v2"), ("safety", "6 benches"),
             ("abstention", "deferral set"), ("redteam", "30 attacks")]
    for i, (a, b) in enumerate(tasks):
        yy = 1.52 - i * 0.30
        _box(ax, 0.10, yy, 1.70, 0.24, a, b)
    _box(ax, 0.10, 0.28, 1.70, 0.20, "smoke (routing only)", bg=PAPER, edge=LINE)
    scorers = [("footage_score", "VQA / count / sev"), ("safety_score", "label match"),
               ("abstention_score", "risk + ECE"), ("redteam_asr", "auto + adjudicated")]
    for i, (a, b) in enumerate(scorers):
        yy = 1.52 - i * 0.30
        _box(ax, 2.20, yy, 1.90, 0.24, a, b, bg=DATA_BG, edge=GOLD)
        _arrow(ax, 1.80, yy + 0.12, 2.20, yy + 0.12)
    _box(ax, 4.50, 0.95, 1.60, 0.60, ".eval logs", "per-sample vectors",
         bg=CARD, edge=LINE)
    for yy, ty in ((1.64, 1.42), (1.34, 1.30), (1.04, 1.18), (0.74, 1.06)):
        _arrow(ax, 4.10, yy, 4.50, ty)
    _box(ax, 6.50, 1.28, 1.55, 0.48, "aggregate", "record / table",
         bg=OUT_BG, edge=TEAL)
    _box(ax, 6.50, 0.62, 1.55, 0.48, "plot scripts", "curves + ASR",
         bg=OUT_BG, edge=TEAL)
    _arrow(ax, 6.10, 1.25, 6.50, 1.52)
    _arrow(ax, 6.10, 1.05, 6.50, 0.86)
    _box(ax, 8.35, 0.95, 1.55, 0.60, "docs/reports", "md + csv + png",
         bg=OUT_BG, edge=TEAL)
    _arrow(ax, 8.05, 1.52, 8.35, 1.25)
    _arrow(ax, 8.05, 0.86, 8.35, 1.05)
    ax.text(9.85, 0.10, "HSEA original", ha="right", va="center",
            fontsize=6, color=MUTED)
    return _save(fig, out_dir, "fig-tech-harness.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for fn in (fig_loop, fig_dataflow, fig_harness):
        total += fn(out).stat().st_size
    assert total <= 300_000, total
    print(f"total new binaries: {total} bytes (budget 300,000)")


if __name__ == "__main__":
    main()
