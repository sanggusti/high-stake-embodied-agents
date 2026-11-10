"""Render original VLM-alignment survey figures (followup/alignment-vlm).

Outputs (committed under docs/alignment-vlm/):
  fig1_alignment_stack.png  - HSEA footage-VLM alignment stack: data -> objective -> gate -> metric
  fig2_grounding_formats.png - grounding output formats compared (boxes vs points vs grid cells)
  fig3_rlvr_rewards.png     - verifiable reward composition for detection/counting/VQA

Deterministic: Agg backend, no randomness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "alignment-vlm"

STAGE_COLORS = ["#dbeafe", "#e0e7ff", "#fae8ff", "#fef3c7"]
GATE_COLOR = "#fee2e2"
EVAL_COLOR = "#dcfce7"


def _box(ax, xy, w, h, text, face, title=None, fontsize=7.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=face, edgecolor="#334155", lw=1.2))
    if title:
        ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top",
                fontsize=fontsize + 0.5, weight="bold", color="#0f172a")
        ax.text(x + w / 2, y + h / 2 - 0.06, text, ha="center", va="center",
                fontsize=fontsize, color="#1e293b", linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color="#1e293b", linespacing=1.5)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))


def fig1_alignment_stack(path: Path):
    """Four-stage footage alignment pipeline with gates and eval outputs."""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("HSEA footage-VLM alignment stack (survey recommendations)",
                 fontsize=11, weight="bold", pad=10)

    stages = [
        ("1. SFT mix\n(Cambrian-7M recipe)", "FloodNet-VQA + LADI v2\n+ DisasterM3-text + AIDERv2\nper-source caps, ratio study", 0.3),
        ("2. Grounding SFT\n(Molmo points)", "DisasterM3 referring\npoints 0-100 + boxes\ncount-by-pointing", 3.3),
        ("3. On-policy DPO\n(OPA-DPO + RLAIF-V)", "open-VLM judges\norig-vs-distorted pairs\n4.8k-scale pilot", 6.3),
        ("4. Vision RLVR\n(VLM-R1 + Visual-RFT)", "GRPO: IoU + EM +\nformat rewards\ncounting / severity", 9.3),
    ]
    for (title, body, x), color in zip(stages, STAGE_COLORS):
        _box(ax, (x, 2.5), 2.4, 1.9, body, color, title=title)
    for x1, x2 in ((2.7, 3.3), (5.7, 6.3), (8.7, 9.3)):
        _arrow(ax, x1, 3.45, x2 - 0.05, 3.45)

    gates = [
        ("G1\nrefusal holds\nvs base", 2.75),
        ("G2\nmisuse refusal\nbenign flat", 5.75),
        ("G3\nembodied safety\n+ calibration", 8.75),
    ]
    for text, x in gates:
        _box(ax, (x, 1.15), 1.5, 1.0, text, GATE_COLOR, fontsize=7)

    _box(ax, (0.3, 0.15), 5.5, 0.75,
         "footage_score vector: vqa_acc / counting_acc+MAE / severity_f1 / grounding_acc",
         EVAL_COLOR, title="Evals (existing)")
    _box(ax, (6.3, 0.15), 5.4, 0.75,
         "new: box-IoU + pointing_acc + HSEA-POPE F1 + hal-rate + abstain-rate",
         EVAL_COLOR, title="Evals (proposed)")
    _arrow(ax, 3.0, 1.15, 3.0, 0.95)
    _arrow(ax, 9.0, 1.15, 9.0, 0.95)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fake_frame(ax, title):
    """Stylized aerial frame: water band, buildings, debris dots."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, weight="bold")
    ax.add_patch(plt.Rectangle((0, 0), 100, 30, facecolor="#bae6fd",
                               edgecolor="none"))
    for (x, y, w, h) in ((10, 55, 22, 16), (42, 62, 18, 14), (68, 50, 24, 18)):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#cbd5e1",
                                   edgecolor="#475569", lw=1))
    for (x, y) in ((20, 15), (45, 20), (60, 12), (80, 22), (35, 8)):
        ax.plot(x, y, "o", ms=4, color="#78350f")
    ax.text(50, 82, "query: flooded houses + debris", ha="center", fontsize=7,
            style="italic", color="#334155")


def fig2_grounding_formats(path: Path):
    """Three grounding output formats on the same stylized frame."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    fig.suptitle("Grounding output formats for footage referring (same frame)",
                 fontsize=11, weight="bold")

    _fake_frame(axes[0], "(a) bbox-as-tokens  [Qwen / PaliGemma]")
    axes[0].add_patch(plt.Rectangle((10, 55), 22, 16, fill=False,
                                    edgecolor="#dc2626", lw=2))
    axes[0].add_patch(plt.Rectangle((68, 50), 24, 18, fill=False,
                                    edgecolor="#dc2626", lw=2))
    axes[0].text(21, 73, "[10,55,32,71]", fontsize=6.5, color="#dc2626",
                 weight="bold")
    axes[0].set_xlabel("text spans or <loc> ids; IoU-verifiable\n"
                         "metric: Acc@IoU0.5 / mAP", fontsize=7.5)

    _fake_frame(axes[1], "(b) pointing  [Molmo / PixMo]")
    pts = [(21, 63), (80, 59), (45, 20), (60, 12)]
    for i, (x, y) in enumerate(pts, 1):
        axes[1].plot(x, y, "o", ms=7, color="#16a34a")
        axes[1].text(x + 2, y + 2, str(i), fontsize=8, weight="bold",
                     color="#14532d")
    axes[1].set_xlabel("plain-text x,y in 0-100; count-by-pointing CoT\n"
                         "metric: point-in-box / MAE", fontsize=7.5)

    _fake_frame(axes[2], "(c) grid-cell  [HSEA today -> box ext.]")
    for gx in (0, 25, 50, 75, 100):
        axes[2].axvline(gx, color="#64748b", lw=0.8, ls="--")
    for gy in (0, 25, 50, 75, 100):
        axes[2].axhline(gy, color="#64748b", lw=0.8, ls="--")
    for (x, y, lab) in ((12.5, 62.5, "B2"), (87.5, 62.5, "B4"),
                        (37.5, 12.5, "D2")):
        axes[2].text(x, y, lab, fontsize=9, weight="bold", color="#7c3aed",
                     ha="center", va="center",
                     bbox=dict(facecolor="white", alpha=0.8, pad=1, lw=0))
    axes[2].set_xlabel("cell ids (B3); coarse but total + deterministic\n"
                         "metric: grounding_acc -> +IoU", fontsize=7.5)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.24, top=0.84,
                        wspace=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig3_rlvr_rewards(path: Path):
    """Verifiable reward composition per footage task family."""
    tasks = ["referring\ngrounding", "counting", "severity\n(yes/no)",
             "presence\nVQA"]
    acc = [0.62, 0.55, 0.71, 0.83]
    conf = [0.18, 0.15, 0.12, 0.07]
    fmt = [0.20, 0.30, 0.17, 0.10]
    rewards = ["R_acc: IoU / EM / EM / EM", "R_conf: calibration",
               "R_format: parse check"]
    colors = ["#60a5fa", "#fbbf24", "#34d399"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.suptitle("Vision-RLVR reward composition mapped to footage tasks",
                 fontsize=11, weight="bold")
    bottom = [0.0] * len(tasks)
    for vals, label, color in zip((acc, conf, fmt), rewards, colors):
        ax.bar(tasks, vals, bottom=bottom, label=label, color=color,
               edgecolor="#334155", lw=1)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, t in enumerate(tasks):
        ax.text(i, 1.02, f"R={bottom[i]:.2f}", ha="center", fontsize=8,
                color="#334155")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("reward weight (illustrative mix)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.5, -0.28,
            "GRPO update on Qwen3-VL-8B / Molmo2-8B: sample N=8, group-normalize, "
            "KL to SFT ref.\nCounting/format weight is highest: FloodNet counting "
            "fails today on verbose non-parseable outputs.",
            ha="center", va="top", fontsize=7.5, color="#334155",
            transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1_alignment_stack(out / "fig1_alignment_stack.png")
    fig2_grounding_formats(out / "fig2_grounding_formats.png")
    fig3_rlvr_rewards(out / "fig3_rlvr_rewards.png")
    print(f"wrote 3 figures to {out}")


if __name__ == "__main__":
    main()
