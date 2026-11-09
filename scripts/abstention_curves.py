"""Abstention curves (#21): risk-coverage + ECE from .eval logs.

Reads one .eval log per model, recomputes the defer-to-human analysis on
the committed cal/eval splits, prints markdown tables, and writes two
PNGs (risk-coverage, calibration).

Example:
    .venv/bin/python scripts/abstention_curves.py \\
        --logs logs/a_abstention_xyz.eval logs/b_abstention_xyz.eval \\
        --labels muse-spark,deepseek-v41-flash \\
        --out-dir docs/reports --prefix issue-21-abstention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hsea.scorers.abstention import (
    aurc,
    expected_calibration_error,
    native_point,
    pick_threshold,
    risk_coverage,
)

TAU_GRID = [i / 100 for i in range(0, 101, 5)]
MAX_RISK = 0.05
SCORER_NAME = "abstention_score"


def load_items(log_path: str) -> list[dict]:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    log = read_eval_log(log_path)
    items = []
    for sample in log.samples or []:
        meta = dict(sample.metadata or {})
        score = (sample.scores or {}).get(SCORER_NAME)
        smeta = dict(getattr(score, "metadata", None) or {})
        pred = smeta.get("abst_pred", "none")
        conf = smeta.get("abst_confidence")
        items.append(
            {
                "id": sample.id,
                "set": meta.get("set"),
                "split": meta.get("split"),
                "label": smeta.get("abst_label", meta.get("label")),
                "comply_pred": pred == "comply",
                "confidence": conf if isinstance(conf, (int, float)) else None,
                "soft": float(smeta.get("abst_soft", 0.0)),
            }
        )
    return items


def summarize(items: list[dict]) -> dict:
    cal = [it for it in items if it["split"] == "cal"]
    evl = [it for it in items if it["split"] == "eval"]
    cal_curve = risk_coverage(cal, TAU_GRID)
    evl_curve = risk_coverage(evl, TAU_GRID)
    tau_star = pick_threshold(cal_curve, MAX_RISK)
    applied = risk_coverage(evl, [tau_star["tau"]])[0]
    ece = expected_calibration_error(evl)
    soft_acc = sum(it["soft"] for it in evl) / len(evl) if evl else 0.0
    return {
        "n_cal": len(cal),
        "n_eval": len(evl),
        "native_cal": native_point(cal),
        "native_eval": native_point(evl),
        "cal_curve": cal_curve,
        "eval_curve": evl_curve,
        "aurc_eval": aurc(evl_curve),
        "tau_star": tau_star,
        "applied_eval": applied,
        "ece": ece,
        "soft_acc_eval": soft_acc,
    }


def summary_table(results: dict[str, dict]) -> str:
    lines = [
        "| model | n_cal/n_eval | native risk | native cov | over-refusal "
        "(native) | soft acc | ECE | tau* (cal risk/cov) | tau* on eval (risk/cov/over-ref) | AURC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in results.items():
        nat = s["native_eval"]
        t = s["tau_star"]
        a = s["applied_eval"]
        lines.append(
            "| " + " | ".join([
                name,
                f"{s['n_cal']}/{s['n_eval']}",
                f"{nat['risk']:.3f}",
                f"{nat['coverage']:.3f}",
                f"{nat['over_refusal']:.3f}",
                f"{s['soft_acc_eval']:.3f}",
                f"{s['ece']['ece']:.3f}",
                f"{t['tau']:.2f} ({t['risk']:.3f}/{t['coverage']:.3f})",
                f"{a['risk']:.3f}/{a['coverage']:.3f}/{a['over_refusal']:.3f}",
                f"{s['aurc_eval']:.3f}",
            ]) + " |"
        )
    return "\n".join(lines) + "\n"


def curve_table(name: str, points: list[dict], taus: list[float]) -> str:
    keep = [p for p in points if any(abs(p["tau"] - t) < 1e-9 for t in taus)]
    lines = [
        f"risk-coverage ({name}, eval):",
        "",
        "| tau | coverage | risk | over-refusal | n_acted/n |",
        "|---|---|---|---|---|",
    ]
    for p in keep:
        lines.append(
            f"| {p['tau']:.2f} | {p['coverage']:.3f} | {p['risk']:.3f} "
            f"| {p['over_refusal']:.3f} | {p['n_acted']:.0f}/{p['n_total']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def reliability_table(name: str, ece: dict) -> str:
    lines = [
        f"reliability ({name}, eval, ECE={ece['ece']:.3f}, "
        f"MCE={ece['mce']:.3f}, n={ece['n']:.0f}, "
        f"missing-conf={ece['n_missing_conf']:.0f}):",
        "",
        "| bin | n | mean conf | acc | gap |",
        "|---|---|---|---|---|",
    ]
    for b in ece["bins"]:
        if b["n"] == 0:
            lines.append(f"| ({b['lo']:.1f},{b['hi']:.1f}] | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| ({b['lo']:.1f},{b['hi']:.1f}] | {b['n']:.0f} | {b['conf']:.3f} "
            f"| {b['acc']:.3f} | {abs(b['acc'] - b['conf']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def plot_risk_coverage(results: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in results.items():
        pts = sorted(s["eval_curve"], key=lambda p: p["coverage"])
        ax.plot(
            [p["coverage"] for p in pts], [p["risk"] for p in pts],
            marker=".", markersize=3, label=f"{name} (AURC {s['aurc_eval']:.3f})",
        )
        nat = s["native_eval"]
        ax.scatter([nat["coverage"]], [nat["risk"]], s=80, marker="o",
                   zorder=5, label=f"{name} native")
        a = s["applied_eval"]
        ax.scatter([a["coverage"]], [a["risk"]], s=80, marker="D",
                   zorder=5, label=f"{name} tau*={s['tau_star']['tau']:.2f}")
    ax.set_xlabel("coverage (fraction acted on)")
    ax.set_ylabel("risk (P(label != comply | acted))")
    ax.set_title("Risk-coverage on abstention eval split (risk + benign)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(results: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, s in results.items():
        bins = [b for b in s["ece"]["bins"] if b["n"] > 0]
        ax.plot(
            [b["conf"] for b in bins], [b["acc"] for b in bins],
            marker="o", label=f"{name} (ECE {s['ece']['ece']:.3f})",
        )
    ax.set_xlabel("mean stated confidence per bin")
    ax.set_ylabel("soft accuracy per bin")
    ax.set_title("Calibration on abstention eval split (risk + benign)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    global MAX_RISK  # noqa: PLW0603
    parser = argparse.ArgumentParser(description="Abstention risk-coverage + ECE.")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--labels", required=True, help="Comma-separated model labels.")
    parser.add_argument("--out-dir", default="docs/reports")
    parser.add_argument("--prefix", default="issue-21-abstention")
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--max-risk", type=float, default=MAX_RISK)
    args = parser.parse_args()
    labels = [label.strip() for label in args.labels.split(",")]
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must match --logs one-to-one")

    MAX_RISK = args.max_risk

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, log_path in zip(labels, args.logs):
        items = load_items(log_path)
        results[name] = summarize(items)
        print(f"{name}: {len(items)} samples from {log_path}", flush=True)

    rc_path = out_dir / f"{args.prefix}-risk-coverage.png"
    cal_path = out_dir / f"{args.prefix}-calibration.png"
    plot_risk_coverage(results, rc_path)
    plot_calibration(results, cal_path)
    print(f"wrote {rc_path}\nwrote {cal_path}")

    parts = [summary_table(results)]
    for name, s in results.items():
        taus = sorted({0.0, 0.25, 0.5, 0.75, 0.9, 1.0, s["tau_star"]["tau"]})
        parts.append(curve_table(name, s["eval_curve"], taus))
        parts.append(reliability_table(name, s["ece"]))
    report = "\n".join(parts)
    print(report)
    if args.md_out:
        Path(args.md_out).write_text(report)
        print(f"wrote {args.md_out}")
    # Machine-readable echo for the run record note.
    print(json.dumps({
        name: {
            "native_risk": round(s["native_eval"]["risk"], 4),
            "native_coverage": round(s["native_eval"]["coverage"], 4),
            "over_refusal": round(s["native_eval"]["over_refusal"], 4),
            "ece": round(s["ece"]["ece"], 4),
            "tau_star": round(s["tau_star"]["tau"], 4),
        }
        for name, s in results.items()
    }))


if __name__ == "__main__":
    main()
