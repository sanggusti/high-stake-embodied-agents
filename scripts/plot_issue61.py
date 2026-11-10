"""Storytelling plots for recomputed metrics (#61, phase 2).

Reads docs/reports/issue-61-recompute/*.json (written by
scripts/recompute_metrics.py) and renders claim-titled PNGs with
Wilson whiskers, reference lines, and annotated deltas. House style:
matplotlib Agg, 150 dpi, traced captions in the phase 2 report.

Example:
    .venv/bin/python scripts/plot_issue61.py \\
        --data docs/reports/issue-61-recompute --out-dir docs/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

COLORS = {
    "musespark": "#4C78A8",
    "v41": "#F58518",
    "v41flash": "#F58518",
    "qwen3vl": "#54A24B",
    "mockllm": "#79706E",
    "mock": "#79706E",
}
MODEL_LABEL = {
    "musespark": "muse-spark",
    "v41": "v41-flash",
    "v41flash": "v41-flash",
    "qwen3vl": "Qwen3-VL-8B",
    "mockllm": "mockllm",
    "mock": "mockllm",
}


def _load(data: Path, track: str) -> dict:
    return json.loads((data / f"{track}.json").read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def _save(fig, path: Path, footer: str) -> None:
    fig.text(0.01, 0.005, footer, fontsize=6.5, color="#555555",
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def _eb(ax, x, rate, lo, hi, color, label=None, width=0.35):
    ax.bar(x, rate, width=width, color=color, label=label, zorder=3)
    ax.errorbar([x], [rate], yerr=[[rate - lo], [hi - rate]],
                fmt="none", ecolor="black", capsize=3, zorder=4)


# --- footage ----------------------------------------------------------------


def plot_footage_f1(ft: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, corpus in zip(axes, ("floodnet-vqa", "ladi-v2")):
        block = ft["corpora"][corpus]
        groups = ["composite"] + [k for k in ("vqa", "counting", "severity")
                                  if any(k in m["kinds"]
                                         for m in block["models"].values())]
        models = [m for m in ("musespark", "qwen3vl", "mockllm")
                  if m in block["models"]]
        xs = range(len(groups))
        for j, model in enumerate(models):
            m = block["models"][model]
            off = (j - 1) * 0.24
            for i, g in enumerate(groups):
                cell = m["composite"] if g == "composite" else m["kinds"].get(g)
                if not cell:
                    continue
                _eb(ax, i + off, cell["rate"], cell["wilson_lo"],
                    cell["wilson_hi"], COLORS[model],
                    label=MODEL_LABEL[model] if i == 0 else None, width=0.22)
                ax.text(i + off, min(1.02, cell["wilson_hi"] + 0.03),
                        f"n={cell['n']:.0f}", ha="center", fontsize=7)
        for i, g in enumerate(groups):
            if g == "counting":
                continue  # exact-int chance is near 0, not 0.5
            ax.hlines(0.5, i - 0.45, i + 0.45, color="black",
                      linestyle="--", linewidth=1.2,
                      label="chance (binary EM)" if i == 0 else None)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("accuracy (Wilson 95%)")
        d = block["paired_musespark_minus_qwen"]
        ax.set_title(f"{corpus}: paired delta "
                     f"{d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}], "
                     f"p={d['p_value']:.2f}")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Footage: sliced accuracy with Wilson intervals (F1)",
                 fontweight="bold")
    _save(fig, out / "issue-61-footage-f1.png",
          "trace: recompute footage.json from logs VJtw/iFkj (muse) + SBB6/kDnU (qwen); seed 61")


def plot_footage_f2(ft: dict, out: Path) -> None:
    slices = ft["fine_slices"]
    for key, title, fname in (
            ("floodnet_question_type", "FloodNet by question type",
             "issue-61-footage-f2-floodnet.png"),
            ("ladi_column", "LADI by column", "issue-61-footage-f2-ladi.png")):
        rows = slices[key]
        order = sorted(rows,
                       key=lambda q: abs(rows[q]["musespark"]["rate"]
                                         - rows[q]["qwen3vl"]["rate"]),
                       reverse=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(order) + 1.5)))
        for i, q in enumerate(order):
            for j, model in enumerate(("musespark", "qwen3vl")):
                cell = rows[q][model]
                y = len(order) - 1 - i + (0.15 if j == 0 else -0.15)
                faded = 0.45 if cell["n"] < 2 else 1.0
                ax.errorbar([cell["rate"]], [y],
                            xerr=[[(cell["rate"] - cell["wilson_lo"])],
                                  [(cell["wilson_hi"] - cell["rate"])]],
                            fmt="o", color=COLORS[model], alpha=faded,
                            capsize=3, markersize=5,
                            label=MODEL_LABEL[model] if i == 0 else None)
                ax.text(1.02, y, f"n={cell['n']:.0f}", fontsize=7, va="center")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(list(reversed(order)), fontsize=8)
        ax.set_xlim(-0.05, 1.25)
        ax.set_xlabel("accuracy (Wilson 95%; faded rows have n=1)")
        ax.set_title(f"{title}: the gap lives in named slices (F2)",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        _save(fig, out / fname,
              "trace: recompute footage.json fine_slices; faded = n=1, full-width intervals shown")


def plot_footage_f3(ft: dict, out: Path) -> None:
    diag = ft["counting_diagnosis"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for model, marker in (("musespark", "o"), ("qwen3vl", "s")):
        pts = diag[model]["points"]
        ok = [p for p in pts if p["format_ok"]]
        bad = [p for p in pts if not p["format_ok"]]
        for subset, ms, mew in ((ok, marker, 1), (bad, "X", 2)):
            if not subset:
                continue
            axes[0].scatter([p["output_tokens"] for p in subset],
                            [p["mae"] if p["mae"] is not None else -0.5
                             for p in subset],
                            c=COLORS[model], marker=ms, s=90,
                            label=f"{MODEL_LABEL[model]}"
                                  f"{' format-bad' if subset is bad else ''}")
    axes[0].set_xlabel("output tokens per sample")
    axes[0].set_ylabel("counting MAE (None shown at -0.5)")
    axes[0].set_title("MAE vs verbosity (format-bad marked X)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    # 2x2 table panel.
    axes[1].axis("off")
    table = [["model", "correct/format-ok", "wrong/format-ok", "wrong/format-bad"]]
    for model in ("musespark", "qwen3vl"):
        t = diag[model]["table_2x2"]
        table.append([MODEL_LABEL[model], str(t["correct/format-ok"]),
                      str(t["wrong/format-ok"]), str(t["wrong/format-bad"])])
    tbl = axes[1].table(cellText=table, loc="center", colWidths=[0.25] * 4)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    axes[1].set_title("Counting 2x2 (correct/format-bad is 0 by construction)")
    fig.suptitle("Footage counting: perception error vs verbosity (F3)",
                 fontweight="bold")
    _save(fig, out / "issue-61-footage-f3.png",
          "trace: recompute footage.json counting_diagnosis; flags via extract_int")


def plot_footage_f4(ft: dict, out: Path) -> None:
    yes = ft["yes_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    models = ("musespark", "qwen3vl")
    acc = [ft["corpora"]["ladi-v2"]["models"][m]["kinds"]["vqa"]["rate"]
           for m in models]
    yr = [yes[m]["yes_rate"]["rate"] for m in models]
    xs = range(len(models))
    axes[0].bar([x - 0.2 for x in xs], acc, width=0.35,
                color="#4C78A8", label="vqa accuracy")
    axes[0].bar([x + 0.2 for x in xs], yr, width=0.35,
                color="#E45756", label="yes-rate P(pred yes)")
    for x, a, y in zip(xs, acc, yr):
        axes[0].text(x - 0.2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(x + 0.2, y + 0.02, f"{y:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels([MODEL_LABEL[m] for m in models])
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel("rate")
    axes[0].set_title("Perfect accuracy needs the yes-rate column")
    axes[0].legend(fontsize=8)
    axes[1].axis("off")
    table = [["model", "gold-y/pred-y", "gold-y/pred-n",
              "gold-n/pred-y", "gold-n/pred-n"]]
    for m in models:
        c = yes[m]["severity_confusion"]
        table.append([MODEL_LABEL[m],
                      str(c["gold-True/pred-True"]),
                      str(c["gold-True/pred-False"]),
                      str(c["gold-False/pred-True"]),
                      str(c["gold-False/pred-False"])])
    tbl = axes[1].table(cellText=table, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    axes[1].set_title("Severity confusion (gold x pred damage grade)")
    fig.suptitle("LADI: the perfect 1.000 is bias-checked (F4)",
                 fontweight="bold")
    _save(fig, out / "issue-61-footage-f4.png",
          "trace: recompute footage.json yes_rate; yes via normalize_answer == 'yes'")


# --- serving -----------------------------------------------------------------


def plot_serving_s1(sv: dict, out: Path) -> None:
    runs = sv["runs"]
    order = list(runs)
    labels = [f"{r['track']}/{r['model']}" for r in runs.values()]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    xs = range(len(order))
    tin = [runs[k]["tokens_per_task"]["input"] for k in order]
    tout = [runs[k]["tokens_per_task"]["output"] for k in order]
    trea = [runs[k]["tokens_per_task"]["reasoning"] for k in order]
    axes[0].bar([x - 0.25 for x in xs], tin, width=0.22, color="#4C78A8",
                label="input")
    axes[0].bar(xs, tout, width=0.22, color="#F58518", label="output")
    axes[0].bar([x + 0.25 for x in xs], trea, width=0.22, color="#54A24B",
                label="reasoning")
    axes[0].set_ylabel("tokens per task")
    axes[0].set_title("Tokens per task by run (S1)")
    axes[0].legend(fontsize=8)
    costs = [runs[k]["cost_per_task"] for k in order]
    has_cost = [c is not None for c in costs]
    known = [c for c in costs if c is not None]
    cmax = max(known) if known else 0.001
    axes[1].bar([x for x, h in zip(xs, has_cost) if h], known,
                width=0.5, color="#79706E")
    for x, c, h in zip(xs, costs, has_cost):
        if h:
            axes[1].text(x, c + cmax * 0.04, f"${c:.4f}",
                         ha="center", fontsize=8)
        else:
            axes[1].text(x, 0.0001, "unledgered", ha="center", fontsize=7,
                         rotation=45)
    axes[1].set_ylabel("$/task (ledgered runs only)")
    axes[1].set_title("Cost per task; footage is unledgered (Baseten slug, Modal GPU)")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    fig.suptitle("Serving: the first cost and token numbers (S1)",
                 fontweight="bold")
    _save(fig, out / "issue-61-serving-s1.png",
          "trace: recompute serving.json; costs from modal-spend.md ledger rows")


def plot_serving_s2(sv: dict, out: Path) -> None:
    runs = sv["runs"]
    order = [k for k in runs if k != "redteam_mock"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, key in enumerate(order):
        lat = runs[key]["latency"]
        samples = lat["samples_s"]
        ax.scatter(samples, [i] * len(samples), s=14,
                   color=COLORS[runs[key]["model"]], alpha=0.6)
        ax.scatter([lat["first_s"]], [i], s=90, marker="D",
                   facecolor="none", edgecolor="black", linewidths=1.5,
                   zorder=5)
        ax.plot([lat["median"]], [i], marker="|", color="black",
                markersize=18, markeredgewidth=2)
        ax.plot([lat["p90"]], [i], marker="|", color="red",
                markersize=18, markeredgewidth=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{runs[k]['track']}/{runs[k]['model']}"
                        for k in order], fontsize=8)
    ax.set_xlabel("seconds per sample (diamond = first sample, "
                  "black tick = median, red tick = p90)")
    ax.set_title("Latency distribution per sample (S2)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, out / "issue-61-serving-s2.png",
          "trace: recompute serving.json latency; completed_at minus started_at")


def plot_serving_s3(sv: dict, out: Path) -> None:
    runs = sv["runs"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key in ("safety_musespark", "safety_v41", "abstention_musespark",
                "abstention_v41"):
        r = runs[key]
        pts = r["verbosity_points"]
        toks = [p["output_reasoning_tokens"] for p in pts]
        ax.scatter(toks, [p["score"] for p in pts], s=16, alpha=0.55,
                   color=COLORS[r["model"]],
                   label=f"{r['track']}/{r['model']} "
                         f"(med {sorted(toks)[len(toks)//2]:.0f} tok)")
    ax.set_xscale("log")
    ax.set_xlabel("output + reasoning tokens (log scale)")
    ax.set_ylabel("per-sample score")
    ax.set_title("Reasoning verbosity vs score: extra tokens buy little (S3)",
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "issue-61-serving-s3.png",
          "trace: recompute serving.json verbosity_points; safety + abstention tracks")


# --- safety ------------------------------------------------------------------


def plot_safety_sa1(sa: dict, out: Path) -> None:
    benches = ("harmbench", "airbench", "agentharm", "safeagent",
               "abstain14", "benign")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for j, model in enumerate(("musespark", "v41")):
        off = -0.2 if j == 0 else 0.2
        for i, bench in enumerate(benches):
            cell = sa["benches"][bench][model]["safety_correct"]
            _eb(ax, i + off, cell["rate"], cell["wilson_lo"],
                cell["wilson_hi"], COLORS[model],
                label=MODEL_LABEL[model] if i == 0 else None, width=0.36)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2,
               label="Gate 2 misuse bar (1.0)")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.2,
               label="Gate 1 refusal bar (0.95)")
    ax.axhline(0.80, color="black", linestyle="-.", linewidth=1.2,
               label="Gate 2/3 floors (0.80)")
    for i, bench in enumerate(benches):
        d = sa["benches"][bench]["paired_musespark_minus_v41"]
        ax.text(i, 0.02, f"d={d['delta']:+.2f}\np={d['p_value']:.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("safety_correct (Wilson 95%)")
    ax.set_title("Safety: per-bench correctness clears every gate bar (SA1)",
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "issue-61-safety-sa1.png",
          "trace: recompute safety.json benches; gates from docs/safety-gates.md")


def plot_safety_sa2(sa: dict, out: Path) -> None:
    panels = sa["two_panel"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, key, title, gate in (
            (axes[0], "harm_refusal", "Refusal on harm rows (higher is safer)",
             "Gate 1/2 bars: 0.95 / 1.0"),
            (axes[1], "benign_compliance",
             "Compliance on benign rows (capability ceiling)", "Gate 2/3 floors: 0.85 / 0.90")):
        benches = [b for b in ("harmbench", "airbench", "agentharm",
                               "safeagent", "abstain14", "benign")
                   if any(panels[key][b][m]["n"] > 0 for m in ("musespark", "v41"))]
        for j, model in enumerate(("musespark", "v41")):
            off = -0.2 if j == 0 else 0.2
            for i, bench in enumerate(benches):
                cell = panels[key][bench][model]
                if cell["n"] == 0:
                    continue
                _eb(ax, i + off, cell["rate"], cell["wilson_lo"],
                    cell["wilson_hi"], COLORS[model],
                    label=MODEL_LABEL[model] if i == 0 else None, width=0.36)
                ax.text(i + off, min(1.03, cell["wilson_hi"] + 0.02),
                        f"n={cell['n']:.0f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(benches)))
        ax.set_xticklabels(benches, rotation=15, ha="right")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("rate (Wilson 95%)")
        ax.set_title(f"{title}; {gate}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Safety vs capability: refuses harm, keeps benign (SA2)",
                 fontweight="bold")
    _save(fig, out / "issue-61-safety-sa2.png",
          "trace: recompute safety.json two_panel; ideal is 1.0 on both panels")


def plot_safety_sa3(sa: dict, out: Path) -> None:
    cells = sa["category_cells"]
    order = sorted(cells,
                   key=lambda c: min(cells[c]["musespark"]["rate"],
                                     cells[c]["v41"]["rate"]))
    fig, ax = plt.subplots(figsize=(8, max(6, 0.42 * len(order) + 1.5)))
    models = ("musespark", "v41")
    for i, cell in enumerate(order):
        y = len(order) - 1 - i
        worst = min(cells[cell][m]["rate"] for m in models)
        for j, model in enumerate(models):
            r = cells[cell][model]
            color = plt.cm.Reds(0.15 + 0.75 * (1.0 - r["rate"]))
            ax.add_patch(Rectangle((j, y - 0.45), 1, 0.9, facecolor=color,
                                   edgecolor="white"))
            ax.text(j + 0.5, y, f"{r['rate']:.2f}\nn={r['n']:.0f}",
                    ha="center", va="center", fontsize=7)
        if worst < 1.0:
            ax.add_patch(Rectangle((0, y - 0.45), 2, 0.9, fill=False,
                                   edgecolor="black", linewidth=1.5))
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["muse-spark", "v41-flash"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=7)
    ax.set_title("Safety by category: worst slices ringed (SA3, M3 pointer)",
                 fontweight="bold")
    _save(fig, out / "issue-61-safety-sa3.png",
          "trace: recompute safety.json category_cells; ring = rate below 1.0")


def plot_safety_sa4m2(sa: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, model in enumerate(("musespark", "v41")):
        m2 = sa["models"][model]["m2"]
        xs = [0 + j * 0.25, 1 + j * 0.25, 2 + j * 0.25]
        vals = [m2["precision"], m2["recall"], m2["f_beta"]]
        ax.bar(xs, vals, width=0.22, color=COLORS[model],
               label=MODEL_LABEL[model])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
        ax.text(1 + j * 0.25, 0.5,
                f"TP={m2['tp']:.0f} FP={m2['fp']:.0f} FN={m2['fn']:.0f}",
                ha="center", fontsize=7)
    ax.set_xticks([0.125, 1.125, 2.125])
    ax.set_xticklabels(["precision", "recall", "F2 (recall x4)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("M2 escalation F2: misses cost 4x false alarms (SA4)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "issue-61-safety-sa4m2.png",
          "trace: recompute safety.json models.*.m2; refuse-on-escalate excluded from FN")


# --- abstention ---------------------------------------------------------------


def _wilson_hi(k: int, n: int) -> float:
    import math

    if n <= 0:
        return 1.0
    p = k / n
    z = 1.959963984540054
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return min(1.0, max(p, center + half))


def plot_abstention_ab1(ab: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model in ("musespark", "v41"):
        m = ab["models"][model]
        pts = sorted(m["curve"], key=lambda p: p["coverage"])
        bands = {b["tau"]: b for b in m["bands"]}
        cov = [p["coverage"] for p in pts]
        risk = [p["risk"] for p in pts]
        ax.plot(cov, risk, color=COLORS[model], linewidth=2,
                label=f"{MODEL_LABEL[model]} (AURC {m['aurc']:.3f})")
        ax.fill_between(cov,
                        [bands[p["tau"]]["risk_lo"] for p in pts],
                        [bands[p["tau"]]["risk_hi"] for p in pts],
                        color=COLORS[model], alpha=0.25)
        # Honest width at the native point: bootstrap gives nothing
        # (zero unsafe acts exist to resample), so the risk whisker is
        # the Wilson upper bound on 0/n_acted and the coverage whisker
        # is the bootstrap band at tau 0.
        nat = m["native"]
        tau0 = next(p for p in m["curve"] if p["tau"] == 0.0)
        band0 = bands[0.0]
        risk_hi = _wilson_hi(0, int(tau0["n_acted"]))
        ax.errorbar([nat["coverage"]], [nat["risk"]],
                    xerr=[[(nat["coverage"] - band0["coverage_lo"])],
                          [(band0["coverage_hi"] - nat["coverage"])]],
                    yerr=[[0.0], [risk_hi]],
                    fmt="none", ecolor="black", capsize=4, zorder=4)
        ax.scatter([nat["coverage"]], [nat["risk"]], s=90,
                   color=COLORS[model], edgecolor="black", zorder=5)
        ax.annotate(f"0/{tau0['n_acted']:.0f}, up to {risk_hi:.3f}",
                    (nat["coverage"], risk_hi),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
        a = m["applied"]
        ax.scatter([a["coverage"]], [a["risk"]], s=90, marker="D",
                   color=COLORS[model], edgecolor="black", zorder=5)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2,
               label="5% risk gate")
    ax.set_xlabel("coverage (fraction acted on)")
    ax.set_ylabel("risk (P(unsafe | acted))")
    ax.set_title("Risk 0.000 with honest upper bounds (AB1)",
                 fontweight="bold")
    ax.set_xlim(0.30, 0.65)
    ax.set_ylim(-0.01, 0.20)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, out / "issue-61-abstention-ab1.png",
          "trace: recompute abstention.json curve+bands; n=52 eval, seed 61, 2000 resamples; dot=native, diamond=tau*")


def plot_abstention_ab2(ab: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model in ("musespark", "v41"):
        m = ab["models"][model]
        pts = sorted(m["curve"], key=lambda p: p["coverage"])
        bands = {b["tau"]: b for b in m["bands"]}
        cov = [p["coverage"] for p in pts]
        ovr = [p["over_refusal"] for p in pts]
        ax.plot(cov, ovr, color=COLORS[model], linewidth=2,
                label=MODEL_LABEL[model])
        ax.fill_between(cov,
                        [bands[p["tau"]]["over_refusal_lo"] for p in pts],
                        [bands[p["tau"]]["over_refusal_hi"] for p in pts],
                        color=COLORS[model], alpha=0.25)
        nat = m["native"]
        ax.scatter([nat["coverage"]], [nat["over_refusal"]], s=90,
                   color=COLORS[model], edgecolor="black", zorder=5)
        ax.annotate(f"native over-ref {nat['over_refusal']:.3f}",
                    (nat["coverage"], nat["over_refusal"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.set_xlabel("coverage (fraction acted on)")
    ax.set_ylabel("over-refusal (deferral on comply items)")
    ax.set_title("The live tradeoff: coverage vs over-refusal (AB2)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "issue-61-abstention-ab2.png",
          "trace: recompute abstention.json curve+bands; n=52 eval, seed 61")


def plot_abstention_ab3(ab: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [1, 1.6]})
    for j, model in enumerate(("musespark", "v41")):
        m = ab["models"][model]
        ece = m["ece"]
        bins = ece["bins"]
        populated = [b for b in bins if b["n"] > 0]
        axes[0][j].bar([(b["lo"] + b["hi"]) / 2 for b in populated],
                       [b["n"] for b in populated], width=0.08,
                       color=COLORS[model])
        axes[0][j].set_title(f"{MODEL_LABEL[model]} confidence counts "
                             f"(n={ece['n']:.0f})")
        axes[0][j].set_ylabel("count")
        ax = axes[1][j]
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        for b in populated:
            ax.bar((b["lo"] + b["hi"]) / 2, b["acc"], width=0.08,
                   color=COLORS[model], alpha=0.7)
            ax.plot([(b["lo"] + b["hi"]) / 2] * 2, [b["acc"], b["conf"]],
                    color="red", linewidth=3)
        ax.set_xlabel("mean stated confidence per bin")
        ax.set_ylabel("soft accuracy per bin")
        ax.set_title(f"reliability (ECE {ece['ece']:.3f}, MCE {ece['mce']:.3f}; "
                     f"red = gap)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Calibration: confidence histogram plus gap bars (AB3)",
                 fontweight="bold")
    _save(fig, out / "issue-61-abstention-ab3.png",
          "trace: recompute abstention.json ece bins; 10 equal bins on eval split")


def plot_abstention_ab4(ab: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    models = ("musespark", "v41")
    keys = sorted({k for m in models for k in ab["models"][m]["slices"]})
    worst_key = max(keys, key=lambda k: max(
        ab["models"][m]["slices"][k]["native"]["over_refusal"] for m in models))
    for i, key in enumerate(keys):
        y = len(keys) - 1 - i
        for j, model in enumerate(models):
            s = ab["models"][model]["slices"][key]
            ovr = s["native"]["over_refusal"]
            color = plt.cm.Reds(0.1 + 0.8 * min(1.0, ovr * 2))
            ax.add_patch(Rectangle((j, y - 0.45), 1, 0.9, facecolor=color,
                                   edgecolor="white"))
            ax.text(j + 0.5, y, f"{ovr:.2f}\nn={s['n']}",
                    ha="center", va="center", fontsize=7)
        if key == worst_key:
            ax.add_patch(Rectangle((0, y - 0.45), 2, 0.9, fill=False,
                                   edgecolor="black", linewidth=2))
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["muse-spark", "v41-flash"])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(list(reversed(keys)), fontsize=7)
    ax.set_title("Over-refusal by slice: worst slice ringed (AB4)",
                 fontweight="bold")
    _save(fig, out / "issue-61-abstention-ab4.png",
          "trace: recompute abstention.json slices; ring = max over-refusal slice")


def plot_abstention_ab5m1(ab: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, model in enumerate(("musespark", "v41")):
        m = ab["models"][model]
        nat = m["native"]
        m1 = m["m1_ehwr_at_tau_star"]
        ax.scatter([nat["coverage"]], [nat["risk"]], s=120,
                   color=COLORS[model], marker="o", edgecolor="black",
                   label=f"{MODEL_LABEL[model]} (risk, cov)")
        ax.scatter([nat["coverage"]], [m1["ehwr"]], s=120,
                   color=COLORS[model], marker="D", edgecolor="black",
                   label=f"{MODEL_LABEL[model]} (EHWR, cov)")
        ax.annotate(f"EHWR={m1['ehwr']:.3f} n_acted={m1['n_acted']}",
                    (nat["coverage"], m1["ehwr"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel("coverage at tau*")
    ax.set_ylabel("risk / EHWR at tau*")
    ax.set_title("M1 EHWR point proves the pipeline (degenerate 0) (AB5)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out / "issue-61-abstention-ab5m1.png",
          "trace: recompute abstention.json m1_ehwr_at_tau_star; risk flat 0 so EHWR is 0")


# --- red-team ----------------------------------------------------------------


def plot_redteam_rt1(rt: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    groups = ["jailbreak", "retrieval_injection", "misuse", "OVERALL"]
    models = ("v41", "mock")
    for j, model in enumerate(models):
        m = rt["models"][model]
        off = -0.2 if j == 0 else 0.2
        for i, group in enumerate(groups):
            cell = (m["asr_overall"] if group == "OVERALL"
                    else m["asr_by_family"][group])
            _eb(ax, i + off, cell["rate"], cell["wilson_lo"],
                cell["wilson_hi"], COLORS[model],
                label=MODEL_LABEL[model] if i == 0 else None, width=0.36)
            if cell["rate"] == 0.0:
                ax.scatter([i + off], [0.015], s=60, marker="o",
                           facecolor="white", edgecolor="black", zorder=5)
                ax.text(i + off, cell["wilson_hi"] + 0.03,
                        f"0/{cell['n']:.0f} up to {cell['wilson_hi']:.2f}",
                        ha="center", fontsize=7)
            else:
                ax.text(i + off, cell["wilson_hi"] + 0.03,
                        f"n={cell['n']:.0f}", ha="center", fontsize=7)
    d = rt["paired_v41_minus_mock"]
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["jailbreak", "retrieval\ninjection", "misuse",
                        f"OVERALL\nd={d['delta']:+.2f} p={d['p_value']:.4f}"])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("ASR (Wilson 95%; adjudicated equals automated)")
    ax.set_title("Red-team ASR: 0/10 is consistent with up to 28% (RT1)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "issue-61-redteam-rt1.png",
          "trace: recompute redteam.json; adjudicated = automated (both review rows attack-failed)")


def plot_redteam_rt2(rt: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in (
            (axes[0], "asr_by_severity", "ASR by severity"),
            (axes[1], "asr_by_variant", "ASR by variant")):
        groups = sorted({g for m in ("v41", "mock")
                         for g in rt["models"][m][key]})
        for j, model in enumerate(("v41", "mock")):
            off = -0.2 if j == 0 else 0.2
            for i, group in enumerate(groups):
                cell = rt["models"][model][key][group]
                _eb(ax, i + off, cell["rate"], cell["wilson_lo"],
                    cell["wilson_hi"], COLORS[model],
                    label=MODEL_LABEL[model] if i == 0 else None, width=0.36)
                ax.text(i + off, min(1.18, cell["wilson_hi"] + 0.04),
                        f"n={cell['n']:.0f}", ha="center", fontsize=6)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1.3)
        ax.set_ylabel("ASR (Wilson 95%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Red-team slicing: which attack style is closest (RT2)",
                 fontweight="bold")
    _save(fig, out / "issue-61-redteam-rt2.png",
          "trace: recompute redteam.json asr_by_severity/asr_by_variant")


def plot_redteam_rt3(rt: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    classes = ["refusal", "hedged comply (needs review)",
               "injection flagged, no refusal", "silent non-comply",
               "full comply"]
    for j, model in enumerate(("v41", "mock")):
        tax = rt["models"][model]["taxonomy"]
        n = rt["models"][model]["n"]
        off = -0.2 if j == 0 else 0.2
        for i, cls in enumerate(classes):
            count = tax.get(cls, 0)
            ax.bar(i + off, count, width=0.36, color=COLORS[model],
                   label=MODEL_LABEL[model] if i == 0 else None)
            if count:
                ax.text(i + off, count + 0.3, f"{count}/{n}",
                        ha="center", fontsize=8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in classes], fontsize=7)
    ax.set_ylabel("completions (count)")
    ax.set_title("Failure taxonomy: binary ASR hides the hedges (RT3)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "issue-61-redteam-rt3.png",
          "trace: recompute redteam.json taxonomy; disjoint partition, sums to n")


def plot_redteam_rt5(rt: dict, out: Path) -> None:
    pk = rt.get("pass_k")
    if not pk:
        return
    panels = [("overall", pk["overall"])] + sorted(pk["by_family"].items())
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.6),
                             sharey=True)
    for ax, (name, entry) in zip(axes, panels):
        ks = sorted(int(k) for k in entry["pass_k_adjudicated"])
        passk = [entry["pass_k_adjudicated"][str(k)] for k in ks]
        passatk = [entry["pass_at_k_adjudicated"][str(k)] for k in ks]
        auto = [entry["pass_k_automated"][str(k)] for k in ks]
        ax.plot(ks, passk, color="#4C78A8", linewidth=2.5,
                label="pass^k adjudicated")
        ax.plot(ks, passatk, color="#4C78A8", linestyle=":", linewidth=2,
                label="pass@k adjudicated")
        ax.plot(ks, auto, color="#79706E", linestyle="--", linewidth=1.5,
                label="pass^k automated")
        ax.set_xticks(ks)
        ax.set_xlabel("k")
        ax.set_ylim(0.75, 1.02)
        ax.set_title(f"{name} (n={entry['n_attacks']})")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("P(all k hold / at least one holds)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Reliability over 5 trials: one jailbreak holds 4/5 (RT5)",
                 fontweight="bold")
    _save(fig, out / "issue-61-redteam-rt5.png",
          "trace: recompute redteam.json pass_k; 5 v41 trials, jb-pair-01 confirmed success")


# --- flywheel -----------------------------------------------------------------


def plot_flywheel_fw1(fw: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(fw["sources"])
    stage_a = [fw["sources"][s]["stage_a"][1] for s in names]
    stage_b = [fw["sources"][s]["stage_b"][1] for s in names]
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], stage_a, width=0.36, color="#79706E",
           label="stage A (sourced units)")
    ax.bar([x + 0.2 for x in xs], stage_b, width=0.36, color="#4C78A8",
           label="stage B (kept rows/pairs)")
    for x, s in zip(xs, names):
        src = fw["sources"][s]
        ax.text(x, max(src["stage_a"][1], src["stage_b"][1]) * 1.06,
                f"{src['ratio_kind']} {src['ratio']:.3f}", ha="center",
                fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("count (log scale)")
    ax.set_title("Curation funnel: yields where it filters, expansions where it renders (FW1)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "issue-61-flywheel-fw1.png",
          "trace: recompute flywheel.json sources; manifest.json + issue-7 report tables")


def plot_flywheel_fw2(fw: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ag = fw["agreement"]
    ax.bar([0, 1], [ag["exact_match"], ag["kappa"]], width=0.5,
           color=["#4C78A8", "#54A24B"])
    for x, v, lab in ((0, ag["exact_match"], "exact-match"),
                      (1, ag["kappa"], "Cohen kappa")):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax.text(x, 0.45, f"n={ag['n_double_labeled']}\n{lab}", ha="center",
                fontsize=8, color="white" if v > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rater exact-match", "rater kappa"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("agreement")
    ax.set_title("Ruler check: blinded second pass agrees 20/20 (FW2)",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "issue-61-flywheel-fw2.png",
          "trace: recompute flywheel.json agreement; rater2.json vs scenarios.jsonl gold")


def plot_flywheel_fw3(fw: dict, out: Path) -> None:
    sp = fw["spend"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    classes = sorted(sp["by_class_usd"])
    vals = [sp["by_class_usd"][c] for c in classes]
    axes[0].bar(classes, vals, color=["#4C78A8", "#F58518", "#54A24B",
                                      "#79706E"][:len(classes)])
    for x, v in zip(classes, vals):
        axes[0].text(x, v + 0.08, f"${v:.2f}", ha="center", fontsize=9)
    axes[0].axhline(50.0, color="black", linestyle="--", linewidth=1.2,
                    label="$50 cap")
    axes[0].set_ylabel("USD")
    axes[0].set_title(f"Ledger TOTAL ${sp['total_usd']:.2f}, "
                      f"headroom ${sp['headroom_usd']:.2f}")
    axes[0].legend(fontsize=8)
    axes[1].axis("off")
    table = [["track", "$/sample"]]
    for track, v in sp["per_sample_usd"].items():
        table.append([track, f"${v:.4f}" if track != "footage" else "unledgered"])
    tbl = axes[1].table(cellText=table, loc="center", colWidths=[0.4, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    axes[1].set_title("Eval $ per scored sample")
    fig.suptitle("Spend: small bills against a large cap (FW3)",
                 fontweight="bold")
    _save(fig, out / "issue-61-flywheel-fw3.png",
          "trace: recompute flywheel.json spend; modal-spend.md ledger union")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot #61 recomputed metrics.")
    parser.add_argument("--data", default="docs/reports/issue-61-recompute")
    parser.add_argument("--out-dir", default="docs/reports")
    args = parser.parse_args()
    _style()
    data = Path(args.data)
    out = Path(args.out_dir)
    ft = _load(data, "footage")
    sv = _load(data, "serving")
    sa = _load(data, "safety")
    ab = _load(data, "abstention")
    rt = _load(data, "redteam")
    fw = _load(data, "flywheel")
    plot_footage_f1(ft, out)
    plot_footage_f2(ft, out)
    plot_footage_f3(ft, out)
    plot_footage_f4(ft, out)
    plot_serving_s1(sv, out)
    plot_serving_s2(sv, out)
    plot_serving_s3(sv, out)
    plot_safety_sa1(sa, out)
    plot_safety_sa2(sa, out)
    plot_safety_sa3(sa, out)
    plot_safety_sa4m2(sa, out)
    plot_abstention_ab1(ab, out)
    plot_abstention_ab2(ab, out)
    plot_abstention_ab3(ab, out)
    plot_abstention_ab4(ab, out)
    plot_abstention_ab5m1(ab, out)
    plot_redteam_rt1(rt, out)
    plot_redteam_rt2(rt, out)
    plot_redteam_rt3(rt, out)
    plot_redteam_rt5(rt, out)
    plot_flywheel_fw1(fw, out)
    plot_flywheel_fw2(fw, out)
    plot_flywheel_fw3(fw, out)


if __name__ == "__main__":
    main()
