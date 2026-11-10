"""Render original HSEA post-training method diagrams (matplotlib only).

Outputs (next to this script):
  fig-rlvr-safety.png      RLVR loop with verifiable safety-procedure rewards
  fig-preference-map.png   preference-stage decision map (data regime -> method)
  fig-testtime-verify.png  compute-optimal test-time verification routing

All three are original redraws for HSEA (flywheel-v2 / dispatch /
abstention labels); they explain the recommended methods, not copies of
any published figure. Run:
  MPLCONFIGDIR=/tmp .venv/bin/python docs/alignment-methods-posttraining/render_figs.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

INK = "#1a1a1a"
ACCENT = "#0b5fff"
GOOD = "#0a7a3d"
WARN = "#b25a00"
PANEL = "#f2f5fb"
PANEL_EDGE = "#9db4d8"


def panel(ax, xy, w, h, title, lines, title_color=ACCENT):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=PANEL, edgecolor=PANEL_EDGE, lw=1.5))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top",
            fontsize=11, weight="bold", color=title_color)
    ax.text(x + w / 2, y + h - 0.75, "\n".join(lines), ha="center", va="top",
            fontsize=9, color=INK, linespacing=1.5)


def arrow(ax, a, b, color=INK, style="-", rad=0.0, label=None):
    con = FancyArrowPatch(a, b, arrowstyle="-|>", color=color, lw=1.6,
                          linestyle=style, shrinkA=2, shrinkB=4,
                          connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(con)
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8,
                color=color, style="italic")


def fig_rlvr():
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.set_title("E-RLVR pilot: RLVR with verifiable SAFETY-PROCEDURE rewards (GRPO)",
                 fontsize=13, weight="bold", color=INK, pad=12)
    panel(ax, (0.2, 3.9), 2.6, 1.7, "Prompts (no new labels)",
          ["7,000 dispatch situations", "60 abstention scenarios",
           "vignette rollouts (L2)"])
    panel(ax, (3.3, 3.9), 2.6, 1.7, "Policy: sample group G=8",
          ["SFT+DPO checkpoint", "Qwen3.8-27B-LoRA", "mask observation tokens*"])
    panel(ax, (6.4, 3.9), 2.5, 1.7, "Deterministic verifier",
          ["(a) priority exact match", "(b) abstain/escalate match",
           "(c) confirm-before-irreversible", "(d) typed output parses"],
          title_color=GOOD)
    panel(ax, (9.4, 3.9), 2.4, 1.7, "GRPO update",
          ["group-relative advantage", "no critic, KL to ref", "G=8 → reward/var"],
          title_color=WARN)
    panel(ax, (3.3, 1.3), 5.6, 1.9, "Gates (every run)",
          ["Gate 3: SafeAgentBench + escalation recall + ECE + benign",
           "dispatch holdout + OOD rule-shift gap (§12.2)",
           "must beat E-RFT on same verifier"],
          title_color=GOOD)
    arrow(ax, (2.8, 4.75), (3.3, 4.75))
    arrow(ax, (5.9, 4.75), (6.4, 4.75))
    arrow(ax, (8.9, 4.75), (9.4, 4.75))
    arrow(ax, (10.4, 3.9), (8.3, 3.2), color=WARN, style="--")
    ax.text(9.75, 3.28, "grade; fail → repair", ha="center", va="top",
            fontsize=8, color=WARN, style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    arrow(ax, (6.1, 1.3), (6.1, 0.95), color=GOOD)
    ax.text(6.1, 0.85, "pass → ship checkpoint + report", ha="center",
            va="top", fontsize=9, color=GOOD, weight="bold")
    ax.text(0.2, 0.35, "* Search-R1 trick: no loss on env/search tokens.",
            fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-rlvr-safety.png"), dpi=150)
    plt.close(fig)


def fig_prefmap():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Preference-stage decision map: data regime → method (E-DPO bake-off)",
                 fontsize=13, weight="bold", color=INK, pad=12)
    panel(ax, (4.2, 4.4), 3.6, 1.2, "Start: preference data?",
          ["7k dispatch pairs + refusal pairs"])
    panel(ax, (0.3, 2.5), 2.6, 1.4, "Pairs + ref fits?",
          ["YES → length-norm DPO", "(Tülu-3 default)"], title_color=GOOD)
    panel(ax, (3.2, 2.5), 2.6, 1.4, "Pairs + OOM?",
          ["YES → SimPO", "(ref-free, terse)"], title_color=GOOD)
    panel(ax, (6.1, 2.5), 2.6, 1.4, "Only binary labels?",
          ["YES → KTO", "(L2/abstention)"], title_color=GOOD)
    panel(ax, (9.0, 2.5), 2.6, 1.4, "One stage, ≤7B?",
          ["YES → ORPO", "(edge lanes)"], title_color=WARN)
    panel(ax, (3.2, 0.5), 5.6, 1.4, "All arms graded identically",
          ["dispatch acc (holdout + OOD) · Gate 1 · benign · abstain recall · length"],
          title_color=INK)
    arrow(ax, (5.0, 4.4), (1.6, 3.9))
    arrow(ax, (5.5, 4.4), (4.5, 3.9))
    arrow(ax, (6.5, 4.4), (7.4, 3.9))
    arrow(ax, (7.2, 4.4), (10.3, 3.9))
    for x in (1.6, 4.5, 7.4, 10.3):
        arrow(ax, (x, 2.5), (6.0, 1.9), color="#666666", style=":")
    ax.text(0.3, 0.55, "IPO = 4th arm if DPO overfits templated pairs.",
            fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-preference-map.png"), dpi=150)
    plt.close(fig)


def fig_testtime():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("Compute-optimal test-time verification for dispatch/abstention (E-TTC)",
                 fontsize=13, weight="bold", color=INK, pad=12)
    panel(ax, (0.2, 3.9), 2.4, 1.7, "Prompt in",
          ["dispatch situation", "or abstention case", "stakes tag"])
    panel(ax, (3.1, 3.9), 2.6, 1.7, "Greedy decode",
          ["top-1 + confidence", "cheap: mean-N ≈ 1"])
    panel(ax, (6.2, 3.9), 2.5, 1.7, "Verifier gate",
          ["conf < τ? or stakes=high?", "PRM step scores", "→ escalate if unsure"],
          title_color=WARN)
    panel(ax, (9.2, 3.9), 2.6, 1.7, "Best-of-8 + verify",
          ["rerank by verifier", "report bon_gain (§12.2)", "PRM + rule checks"],
          title_color=GOOD)
    panel(ax, (6.2, 1.3), 2.5, 1.9, "Abstain / escalate",
          ["verifier disagreement", "→ human / commander", "log + fingerprint"],
          title_color=GOOD)
    panel(ax, (9.2, 1.3), 2.6, 1.9, "Answer",
          ["verified dispatch", "typed + cited", "log + fingerprint"])
    arrow(ax, (2.6, 4.75), (3.1, 4.75))
    arrow(ax, (5.7, 4.75), (6.2, 4.75))
    arrow(ax, (8.7, 4.75), (9.2, 4.75), label="verify")
    arrow(ax, (7.45, 3.9), (7.45, 3.2), color=WARN, label="unsure")
    arrow(ax, (10.5, 3.9), (10.5, 3.2), color=GOOD)
    ax.text(0.2, 0.5, "Snell: 4× over fixed BoN · s1: budget-forcing second pass = sequential twin.",
            fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-testtime-verify.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_rlvr()
    fig_prefmap()
    fig_testtime()
    print("wrote 3 figs to", OUT)
