"""Showcase round 2 loop figures (Space-side generation, bench-side data).

Renders three Findings figures from traced bench-pin data (bench main
46f35ec0). Every number below cites its bench artifact in a TRACE
comment. Style matches the Space Emberline palette (see theme.css)
and the bench plot scripts (matplotlib Agg, DejaVu Sans).

Rescued 2026-11-12 from the stranded /tmp original onto this branch;
rendering code is unchanged, only the input/output paths moved to
CLI flags (original defaults were /tmp/bench-46f35ec and the Space
assets checkout).

Outputs (--out-dir, each asserted <50,000 bytes):
  fig-loop-loss.jpg     training loss curves, both concluded loops
  fig-loop-stages.jpg   DPO-stage metrics: base to SFT to DPO per lane
  fig-loop-xmodel.jpg   cross-model comparison: base-to-DPO deltas

Run: MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python \\
       scripts/plot_loop_showcase.py --out-dir /tmp/loopfigs
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

BENCH = Path(__file__).resolve().parents[1]
SPACE_ASSETS = Path.cwd()

EMBER = "#E0531F"
TEAL = "#0E9F8A"
SLATE = "#1F2733"
GOLD = "#E9A13B"
MUTED = "#5F6B7A"
RISK = "#B3261E"
PAPER = "#F7F5F0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})


def save_small(fig: plt.Figure, name: str, rect_bottom: float = 0.0) -> dict:
    """Save width-900 JPEG q82, assert <50KB, return provenance facts."""
    if rect_bottom:
        fig.tight_layout(rect=[0, rect_bottom, 1, 1])
    else:
        fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    img.thumbnail((900, 900))
    out = io.BytesIO()
    used = None
    for q in (82, 80, 72, 65):
        trial = io.BytesIO()
        img.save(trial, format="JPEG", quality=q, optimize=True)
        if len(trial.getvalue()) < 50_000:
            used = q
            out = trial
            break
    assert used is not None, f"{name} cannot fit <50KB"
    data = out.getvalue()
    (SPACE_ASSETS / name).write_bytes(data)
    with Image.open(SPACE_ASSETS / name) as check:
        check.verify()
    with Image.open(SPACE_ASSETS / name) as check2:
        dims = f"{check2.width}x{check2.height}"
    import hashlib

    info = {"bytes": len(data), "dims": dims,
            "sha256": hashlib.sha256(data).hexdigest(), "quality": used}
    print(name, info)
    return info


def fig_loss() -> None:
    """2x2 training loss curves from the committed trainer histories."""
    # TRACE 73: runs/aya73_train_log.json (sft/dpo loss_history);
    #   final losses 0.3987 / 0.0027 per docs/reports/issue-73-loop.md §2.
    # TRACE 74: runs/aya74/training_curves.json (sft/dpo [step, loss]);
    #   final losses 0.371 / 0.008 per docs/reports/issue-74-aya-expanse.md §2.
    h73 = json.loads((BENCH / "runs/aya73_train_log.json").read_text())
    c74 = json.loads((BENCH / "runs/aya74/training_curves.json").read_text())
    s73 = [(r["step"], r["loss"]) for r in h73["sft"]["loss_history"]]
    d73 = [(r["step"], r["loss"]) for r in h73["dpo"]["loss_history"]]
    s74 = [(r[0], r[1]) for r in c74["sft"]]
    d74 = [(r[0], r[1]) for r in c74["dpo"]]
    for label, series in (("73sft", s73), ("73dpo", d73),
                          ("74sft", s74), ("74dpo", d74)):
        print(label, "n=", len(series), "first=", series[0],
              "last=", series[-1])

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    fig.suptitle("Training loss: SFT converges, DPO collapses near zero",
                 fontsize=10, fontweight="bold")
    panels = [
        (axes[0, 0], s73, "Tiny Aya SFT (8,667 rows, final 0.399)", TEAL),
        (axes[0, 1], d73, "Tiny Aya DPO (7,054 pairs, final 0.003)", EMBER),
        (axes[1, 0], s74, "Aya Expanse SFT (8,831 rows, final 0.371)", TEAL),
        (axes[1, 1], d74, "Aya Expanse DPO (7,078 pairs, final 0.008)", EMBER),
    ]
    for ax, series, title, color in panels:
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        ax.plot(xs, ys, color=color, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("train loss")
        ax.set_xlim(0, max(xs))
    save_small(fig, "fig-loop-loss.jpg")


def fig_stages() -> None:
    """Base to SFT to DPO grouped bars on three shared lanes."""
    # TRACE 73: runs/aya73-recompute/before_after.json (base/sft/finetune
    #   rates) and docs/reports/issue-73-loop.md §3.
    # TRACE 74: runs/aya74/det_scores.json checkpoints base/sft/dpo and
    #   docs/reports/issue-74-aya-expanse.md §14 lane table.
    lanes = [
        ("safety", "Safety (78)", (0.218, 0.705, 0.846),
         (0.872, 0.795, 0.795)),
        ("abstention", "Abstention (80)", (0.037, 0.650, 0.362),
         (0.225, 0.863, 0.775)),
        ("asr", "Red-team ASR, lower better (30)",
         (0.633, 0.533, 0.833), (0.767, 0.333, 0.367)),
    ]
    # 73 abstention is soft accuracy, 74 abstention is exact: labeled.
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.0))
    fig.suptitle("Per-stage metrics: SFT does most of the work",
                 fontsize=10, fontweight="bold")
    ckpts = ["base", "SFT", "DPO"]
    colors = [MUTED, TEAL, EMBER]
    for ax, (key, title, v73, v74) in zip(axes, lanes):
        xs = [0, 1, 2, 4, 5, 6]
        vals = list(v73) + list(v74)
        bars = ax.bar(xs, vals, color=colors * 2, width=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["73\nbase", "73\nSFT", "73\nDPO",
                            "74\nbase", "74\nSFT", "74\nDPO"], fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("rate")
        ax.axvline(3, color="#E4E1D8", linewidth=1)
    fig.text(0.5, 0.02,
             "73 abstention is soft accuracy, 74 abstention is exact; "
             "73 safety is label-match, 74 safety is exact.",
             ha="center", fontsize=7, color=MUTED)
    save_small(fig, "fig-loop-stages.jpg", rect_bottom=0.14)


def fig_xmodel() -> None:
    """Base-to-DPO deltas per shared lane, both concluded loops."""
    # TRACE: deltas from runs/aya73-recompute/before_after.json and
    #   runs/aya74/recompute.json; dispatch T5 tallies from
    #   docs/reports/issue-73-loop.md §12 and issue-74-aya-expanse.md §6;
    #   Qwen #8 has no numbers (in flight per docs/reports/issue-8-plan.md).
    lanes = [
        ("safety (78)", 0.628, -0.077),
        ("refusal gate (26)", 0.692, -0.115),
        ("benign comply (10)", 0.800, 0.000),
        ("abstention (80)", 0.325, 0.550),
        ("red-team ASR (30)*", 0.200, -0.400),
        ("footage floor", -0.017, 0.531),
    ]
    labels = [row[0] for row in lanes][::-1]
    d73 = [row[1] for row in lanes][::-1]
    d74 = [row[2] for row in lanes][::-1]
    import numpy as np

    y = np.arange(len(lanes))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(y + 0.2, d73, height=0.38, color=TEAL, label="Tiny Aya #73")
    ax.barh(y - 0.2, d74, height=0.38, color=EMBER, label="Aya Expanse #74")
    for yy, a, b in zip(y, d73, d74):
        ax.text(a + (0.02 if a >= 0 else -0.02), yy + 0.2, f"{a:+.3f}",
                ha="left" if a >= 0 else "right", va="center", fontsize=7)
        ax.text(b + (0.02 if b >= 0 else -0.02), yy - 0.2, f"{b:+.3f}",
                ha="left" if b >= 0 else "right", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DPO minus base (rate points)")
    ax.set_title("Cross-model deltas: same protocol, different prices",
                 fontsize=10, fontweight="bold")
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.02, 0.01,
             "* ASR is lower-better, so +0.200 on #73 is a regression. "
             "Qwen3.8-27B #8 in flight: no numbers yet, slot reserved.",
             ha="left", fontsize=7, color=MUTED)
    save_small(fig, "fig-loop-xmodel.jpg")


def main() -> None:
    global BENCH, SPACE_ASSETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=BENCH)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    BENCH = args.bench_dir
    SPACE_ASSETS = args.out_dir
    SPACE_ASSETS.mkdir(parents=True, exist_ok=True)
    fig_loss()
    fig_stages()
    fig_xmodel()
    print("loop figures OK")


if __name__ == "__main__":
    main()
