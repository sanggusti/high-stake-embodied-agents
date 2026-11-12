"""Render the why-stacks Technical-details diagrams (Space why-section rework).

Outputs (each asserted <50,000 bytes for the Space asset budget):
  fig-tech-why.jpg    stage -> failure-without-it -> gate-that-proves-it
  fig-tech-order.jpg  order chain with skip costs plus the evals band

JPEG via the bench PIL recipe (the PNG originals exceed the 50 KB
Space cap at this text density; sibling loop figures already ship
as JPEG for the same reason).

Palette and helpers match scripts/plot_tech_details.py (Space Emberline
theme). Deterministic: fixed layout, DejaVu Sans (matplotlib bundled).
Arrow endpoints are computed from box edges (never hand-placed).

Data sources (numbers are labels, not computed; see README-why-stacks.md):
  docs/reports/issue-73-loop.md (bench origin/main 46f35ec)
  docs/reports/issue-74-aya-expanse.md (bench origin/main 46f35ec)
  docs/reports/issue-62-meta-judge.md (bench origin/main 46f35ec)
  docs/reports/issue-7-flywheel-v2-curation.md (8,673 / 7,000 / 250)
  docs/safety-gates.md (Gate 1/2/3 bars)

Usage:
  .venv/bin/python scripts/plot_why_stacks.py --out-dir /tmp/whyfigs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

PAPER = "#F7F5F0"
CARD = "#FFFFFF"
INK = "#14181F"
INK2 = "#1F2733"
EMBER = "#E0531F"
GOLD = "#E9A13B"
TEAL = "#0E9F8A"
MUTED = "#5F6B7A"
LINE = "#E4E1D8"
FAIL_BG = "#FDEEE4"
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
    return {"x": x, "y": y, "w": w, "h": h,
            "cx": x + w / 2, "cy": y + h / 2,
            "left": x, "right": x + w, "top": y + h, "bottom": y}


def _arrow(ax, x1, y1, x2, y2, color=INK2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                        shrinkA=1, shrinkB=2),
    )


def _save(fig, out_dir: Path, name: str) -> Path:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER, dpi=100)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    img.thumbnail((900, 900))
    path = out_dir / name
    used = None
    for q in (82, 80, 72, 65):
        trial = io.BytesIO()
        img.save(trial, format="JPEG", quality=q, optimize=True)
        if len(trial.getvalue()) < MAX_BYTES:
            used = q
            path.write_bytes(trial.getvalue())
            break
    assert used is not None, f"{name} cannot fit <50KB"
    with Image.open(path) as check:
        check.verify()
    with Image.open(path) as check2:
        dims = f"{check2.width}x{check2.height}"
    print(f"{name}: {path.stat().st_size} bytes q{used} {dims} OK")
    return path


def fig_why(out_dir: Path) -> Path:
    """Stage -> failure without it -> gate that proves it, five rows."""
    fig, ax = plt.subplots(figsize=(9.0, 4.1), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.text(5, 2.44, "Why these stacks: each stage earns its place",
            ha="center", va="center", fontsize=10, weight="bold", color=INK)
    ax.text(5, 2.28, "stage hands a checked artifact onward; the gate prints the verdict",
            ha="center", va="center", fontsize=7, color=MUTED)
    for cx, head in ((1.00, "Stage"), (4.05, "Failure without it"),
                     (8.05, "Gate that proves it")):
        ax.text(cx, 2.19, head, ha="center", va="center",
                fontsize=7, weight="bold", color=MUTED)
    rows = [
        ("CuratorKIT", "pins + manifest",
         "leakage replaces learning", "unpinned rows train",
         "verify-only + held-out", "8,673 / 7,000 / 250"),
        ("AlignTune", "SFT then DPO",
         "no contract to rank", "base safety 0.218",
         "safety 0.218 to 0.846", "#73 SFT peak, DPO extends"),
        ("SafeTune", "audit + repair",
         "drift ships silently", "Qi et al. warning",
         "ASR 0.533 to 0.833 caught", "Gate 2 OPEN both loops"),
        ("AgentTune", "RLVR + GRPO",
         "dispatch tax stays", "hedged actions win",
         "T5 5-18 repaid 20-2", "#74 DPO to repair"),
        ("Evals + judges", "Inspect + LLM",
         "artifacts read as behavior", "footer flips",
         "26/26 overturn; 31/40 tie", "#74 T2; #62 calib"),
    ]
    y = 1.86
    for stage, ssub, fail, fsub, gate, gsub in rows:
        s = _box(ax, 0.10, y, 1.80, 0.24, stage, ssub,
                 bg=DATA_BG, edge=GOLD)
        f = _box(ax, 2.45, y, 3.20, 0.24, fail, fsub,
                 bg=FAIL_BG, edge=EMBER)
        g = _box(ax, 6.20, y, 3.70, 0.24, gate, gsub,
                 bg=OUT_BG, edge=TEAL)
        _arrow(ax, s["right"], s["cy"], f["left"], f["cy"])
        _arrow(ax, f["right"], f["cy"], g["left"], g["cy"])
        y -= 0.30
    ax.text(9.85, 0.10, "HSEA original", ha="right", va="center",
            fontsize=6, color=MUTED)
    return _save(fig, out_dir, "fig-tech-why.jpg")


def fig_order(out_dir: Path) -> Path:
    """Order chain: skip any stage and the named cost lands below it."""
    fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.text(5, 2.32, "Order matters: each stage hands the next a checked artifact",
            ha="center", va="center", fontsize=10, weight="bold", color=INK)
    ax.text(5, 2.14, "skip costs observed in loops #73 and #74 (bench origin/main 46f35ec)",
            ha="center", va="center", fontsize=7, color=MUTED)
    chain = [
        ("Curate", "pinned export"),
        ("AlignTune", "taught contract"),
        ("SafeTune", "repaired ckpt"),
        ("AgentTune", "dispatch policy"),
    ]
    boxes = []
    for i, (main, sub) in enumerate(chain):
        x = 0.30 + i * 2.45
        bg, edge = (OUT_BG, TEAL) if i == 3 else (
            (DATA_BG, GOLD) if i == 0 else (CARD, LINE))
        boxes.append(_box(ax, x, 1.42, 1.90, 0.46, main, sub,
                          bg=bg, edge=edge))
    for left, right in zip(boxes, boxes[1:]):
        _arrow(ax, left["right"], left["cy"], right["left"], right["cy"])
    skips = [
        ("skip SFT", "no contract to rank"),
        ("skip audit", "ASR 0.533 to 0.833 ships"),
        ("skip RLVR", "dispatch tax 5-18 stays"),
    ]
    for i, (main, sub) in enumerate(skips):
        gap = (boxes[i]["right"] + boxes[i + 1]["left"]) / 2
        s = _box(ax, gap - 1.20, 0.78, 2.40, 0.40, main, sub,
                 bg=FAIL_BG, edge=EMBER)
        ax.plot([gap, gap], [1.42, s["top"]], color=LINE, lw=1.0,
                linestyle=(0, (2, 2)))
    band = _box(ax, 0.30, 0.16, 9.40, 0.38,
                "Inspect evals + LLM judges grade every stage",
                "deterministic verdicts first, reasoned judgment where flat",
                bg=OUT_BG, edge=TEAL)
    _ = band
    ax.text(9.85, 0.02, "HSEA original", ha="right", va="center",
            fontsize=6, color=MUTED)
    return _save(fig, out_dir, "fig-tech-order.jpg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for fn in (fig_why, fig_order):
        total += fn(out).stat().st_size
    assert total <= 300_000, total
    print(f"total new binaries: {total} bytes (budget 300,000)")


if __name__ == "__main__":
    main()
