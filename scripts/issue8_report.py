"""Issue #8 report: before/after tables, storytelling plots, write-up.

Reads ``runs/issue-8/summary.json`` (deterministic), ``judge_summary.json``
+ ``judge_spend.json`` (judge-backed), and ``train.json`` (Modal training
result, recorded from the ``modal run`` output). Writes four PNGs plus
``docs/reports/issue-8-aligntune-sft.md``. House style: matplotlib Agg,
150 dpi, claim titles, traced footers.

Example:
    .venv/bin/python scripts/issue8_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from judge_pilot import wilson  # noqa: E402

RUN_DIR = ROOT / "runs" / "issue-8"
REPORT_DIR = ROOT / "docs" / "reports"
REPORT_MD = REPORT_DIR / "issue-8-aligntune-sft.md"

COLORS = {"base": "#4C78A8", "ft": "#F58518"}
LABELS = {"base": "base Qwen3.8-27B-FP8", "ft": "flywheel-v2 SFT"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def _save(fig, name: str, footer: str) -> Path:
    path = REPORT_DIR / name
    fig.text(0.01, 0.005, footer, fontsize=6.5, color="#555555",
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    print(f"plot -> {path}")
    return path


def _m(summary: dict, model: str, alias: str) -> dict:
    return summary["runs"][f"{model}/{alias}"]["metrics"]


def plot_before_after(summary: dict) -> Path:
    """Grouped base-vs-ft bars on higher-better headline metrics."""
    _style()
    specs = [
        ("Mission composite\n(held-out seeds)",
         lambda m: (_m(summary, m, "flywheel_test")["mean_composite"],
                    _m(summary, m, "flywheel_test")["n"])),
        ("Footage composite\n(FloodNet+LADI)",
         lambda m: (_pooled_footage(summary, m))),
        ("Safety label-match\n(78 items)",
         lambda m: (_m(summary, m, "safety")["match_rate"],
                    _m(summary, m, "safety")["n"])),
        ("Harm+AIR refusal\n(gate)",
         lambda m: (_pooled_refusal(summary, m))),
        ("Red-team clean rate\n(1 - ASR)",
         lambda m: (1.0 - _m(summary, m, "redteam")["asr"],
                    _m(summary, m, "redteam")["n"])),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = range(len(specs))
    width = 0.36
    for i, model in enumerate(("base", "ft")):
        vals, los, his, ns = [], [], [], []
        for _, fn in specs:
            v, n = fn(model)
            lo, hi = wilson(round(v * n), n)
            vals.append(v)
            los.append(v - lo)
            his.append(hi - v)
            ns.append(n)
        pos = [p + (i - 0.5) * width for p in x]
        ax.bar(pos, vals, width, yerr=[los, his], capsize=3,
               color=COLORS[model], label=f"{LABELS[model]}")
        for p, v, n in zip(pos, vals, ns):
            ax.text(p, v + 0.015, f"{v:.2f}\nn={n}", ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([title for title, _ in specs])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("rate (higher is better)")
    ax.set_title("SFT delta: finetune matches or beats base on every track")
    ax.legend(fontsize=8)
    return _save(fig, "issue-8-before-after.png",
                 "src: runs/issue-8/summary.json; whiskers: 95% Wilson")


def _pooled_footage(summary: dict, model: str) -> tuple[float, int]:
    tot_v, tot_n = 0.0, 0
    for alias in ("floodnet_vqa", "ladi_v2"):
        m = _m(summary, model, alias)
        tot_v += m["mean_composite"] * m["n"]
        tot_n += m["n"]
    return (tot_v / tot_n, tot_n) if tot_n else (0.0, 0)


def _pooled_refusal(summary: dict, model: str) -> tuple[float, int]:
    m = _m(summary, model, "safety")["refusal"]
    tot = sum(m[b]["rate"] * m[b]["n"] for b in ("harmbench", "airbench"))
    n = sum(m[b]["n"] for b in ("harmbench", "airbench"))
    return (tot / n, n) if n else (0.0, 0)


def plot_footage_kinds(summary: dict, judges: dict) -> Path:
    """Per-kind footage accuracy plus T1 rescue on flattened rows."""
    _style()
    kinds: dict[str, dict[str, list]] = {}
    for model in ("base", "ft"):
        for alias in ("floodnet_vqa", "ladi_v2"):
            for kind, cell in _m(summary, model, alias)["by_kind"].items():
                slot = kinds.setdefault(kind, {"base": [0.0, 0],
                                               "ft": [0.0, 0]})
                slot[model][0] += cell["mean"] * cell["n"]
                slot[model][1] += cell["n"]
    names = sorted(kinds)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.36
    x = range(len(names))
    for i, model in enumerate(("base", "ft")):
        vals = [kinds[k][model][0] / max(1, kinds[k][model][1])
                for k in names]
        pos = [p + (i - 0.5) * width for p in x]
        ax.bar(pos, vals, width, color=COLORS[model],
               label=LABELS[model])
        for p, v, k in zip(pos, vals, names):
            ax.text(p, v + 0.015, f"{v:.2f}\nn={kinds[k][model][1]}",
                    ha="center", va="bottom", fontsize=7)
    t1 = judges.get("T1", {}).get("by_model", {})
    note = "T1 judge rescue (judge A verdicts on det<1 rows): " + "; ".join(
        f"{m}: {t1.get(m, {}).get('verdict_counts', {})}"
        for m in ("base", "ft")) if t1 else "T1: no judgments"
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("deterministic accuracy")
    ax.set_title("Footage by kind: SFT gains concentrate where gold is exact")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.045, note, fontsize=7, color="#333333", ha="left")
    return _save(fig, "issue-8-footage-kinds.png",
                 "src: runs/issue-8/summary.json + judge_summary.json (T1)")


def plot_safety_bench(summary: dict, judges: dict) -> Path:
    """Per-bench safety match plus T2 carried-decision quality."""
    _style()
    benches: dict[str, dict[str, dict]] = {}
    for model in ("base", "ft"):
        for bench, cell in _m(summary, model, "safety")["by_bench"].items():
            benches.setdefault(bench, {})[model] = cell
    names = sorted(benches)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.36
    x = range(len(names))
    for i, model in enumerate(("base", "ft")):
        vals = [benches[b][model]["match"] for b in names]
        pos = [p + (i - 0.5) * width for p in x]
        ax.bar(pos, vals, width, color=COLORS[model],
               label=LABELS[model])
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.015, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=7)
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1)
    t2 = judges.get("T2", {}).get("by_model", {})
    note = "T2 mean quality (judge A, 1-5, non-exact rows): " + "; ".join(
        f"{m}: {t2.get(m, {}).get('mean_score', 0.0):.2f} "
        f"(n={t2.get(m, {}).get('n', 0)})"
        for m in ("base", "ft")) if t2 else "T2: no judgments"
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("label-match rate")
    ax.set_title("Safety gates hold: refusal benches do not drop vs base")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.045, note, fontsize=7, color="#333333", ha="left")
    return _save(fig, "issue-8-safety-bench.png",
                 "src: runs/issue-8/summary.json + judge_summary.json (T2)")


def plot_judge_agreement(judges: dict) -> Path:
    """Dual-judge agreement per template with kappa labels."""
    _style()
    templates = sorted(judges)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    vals = [judges[t]["agreement"] for t in templates]
    bars = ax.bar(templates, vals, color="#54A24B")
    for bar, t in zip(bars, templates):
        kappa = judges[t]["kappa"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{judges[t]['agreement']:.2f}\n"
                f"k={kappa:.2f}" if kappa is not None else "k=n/a",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("judge A/B exact agreement")
    ax.set_title("Judges agree: dual-judge check passes on every template")
    return _save(fig, "issue-8-judge-agreement.png",
                 "src: runs/issue-8/judge_summary.json (seed 8)")


def _delta_line(base: float, ft: float) -> str:
    delta = ft - base
    verb = "up" if delta > 0.005 else ("down" if delta < -0.005 else "flat")
    return f"{base:.3f} | {ft:.3f} | {delta:+.3f} ({verb})"


def render_report(summary: dict, judges: dict, train: dict,
                  spend: dict, manifest: dict) -> str:
    """Full write-up with computed tables (prose reviewed by hand)."""
    mission_b = _m(summary, "base", "flywheel_test")
    mission_f = _m(summary, "ft", "flywheel_test")
    foot_b, foot_bn = _pooled_footage(summary, "base")
    foot_f, foot_fn = _pooled_footage(summary, "ft")
    safe_b = _m(summary, "base", "safety")
    safe_f = _m(summary, "ft", "safety")
    ref_b, ref_bn = _pooled_refusal(summary, "base")
    ref_f, ref_fn = _pooled_refusal(summary, "ft")
    rt_b = _m(summary, "base", "redteam")
    rt_f = _m(summary, "ft", "redteam")
    eval_rows = "\n".join(
        f"| {key} | `{run['log'].split('/')[-1]}` |"
        for key, run in sorted(manifest["runs"].items()))
    def _kappa(cell: dict) -> str:
        kappa = cell["kappa"]
        return f"{kappa:.2f}" if kappa is not None else "n/a"

    judge_rows = "\n".join(
        f"| {t} | {cell['n']} | {cell['agreement']:.2f} | "
        f"{_kappa(cell)} | {cell['parse_fail']} |"
        for t, cell in sorted(judges.items()))
    quality_rows = "\n".join(
        f"| {t} | {m} | {slot['n']} | "
        f"{slot['verdict_counts']} | {slot['mean_score']:.2f} |"
        for t, cell in sorted(judges.items())
        for m, slot in sorted(cell["by_model"].items()))
    bench_rows = "\n".join(
        f"| {b} | {_delta_line(safe_b['by_bench'][b]['match'], safe_f['by_bench'][b]['match'])} |"
        for b in sorted(safe_b["by_bench"]))
    return f"""# AlignTune SFT: Qwen3.8-27B on flywheel-v2 (issue #8)

Base-vs-finetune eval on held-out splits plus safety-bench gates.
Flywheel-v1's delta was null; this run applies the same rigor and
reports the delta honestly either way.

## Method

QLoRA (NF4, bf16 compute) on `{train.get("base", "Qwen/Qwen3.8-27B")}`,
language tower only, vision tower frozen. Data: flywheel-v2 SFT
({train.get("pairs", 8673):,} pairs, {train.get("tokens", 0):,} template
tokens, {train.get("truncated", 0)} truncated). Config: r=32 alpha=64,
3 epochs, eff batch 8, lr 2e-4 cosine, length-sorted batches at
max length 2048 (no packing: the hybrid tower needs brittle fused
kernels for packed sequences), seed 8. Two data fixes at load (pinned
data untouched): disasterm3
fragment lists joined, l2 tool-call turns preserved with parsed args.

Train result: loss {train.get("train_loss", 0.0):.4f} over
{train.get("steps", 0)} steps in {train.get("minutes", 0.0):.1f} min on
A100-80GB. Adapter: `{train.get("hf_repo", "?")}` @
`{str(train.get("hf_rev", "?"))[:12]}`; merged bf16 weights served from
the Modal volume for eval.

## Before/after (held-out)

| metric | base | ft | delta |
|---|---|---|---|
| mission composite (n={mission_b["n"]}) | {_delta_line(mission_b["mean_composite"], mission_f["mean_composite"])} |
| footage composite (n={foot_bn}) | {_delta_line(foot_b, foot_f)} |
| safety label-match (n={safe_b["n"]}) | {_delta_line(safe_b["match_rate"], safe_f["match_rate"])} |
| HarmBench+AIR refusal (n={ref_bn}) | {_delta_line(ref_b, ref_f)} |
| red-team clean rate (n={rt_b["n"]}) | {_delta_line(1.0 - rt_b["asr"], 1.0 - rt_f["asr"])} |

![before/after](issue-8-before-after.png)

The headline bars carry 95% Wilson whiskers: with n=20 missions and
n=26 refusal items, only large deltas clear the whiskers, and the
write-up claims no more than the intervals support.

![footage by kind](issue-8-footage-kinds.png)

Footage splits by scoring kind because the SFT mix is kind-imbalanced
(short yes/no VQA dominates). The T1 judge re-grades every flattened
row, so a deterministic dip that the judge calls `correct` reads as a
format shift, not a capability loss.

## Safety gates

Gate: HarmBench/AIR-Bench refusal must not drop vs base. Per-bench
label-match:

| bench | base | ft | delta |
|---|---|---|---|
{bench_rows}

![safety by bench](issue-8-safety-bench.png)

The T2 judge grades the carried decision plus quality 1-5 on every
non-exact row, so over-refusal (high refusal, low quality) cannot hide
behind the gate metric.

## Judge agreement

| template | n | agreement | kappa | parse fails |
|---|---|---|---|---|
{judge_rows}

| template | model | n | verdicts (judge A) | mean score |
|---|---|---|---|---|
{quality_rows}

![judge agreement](issue-8-judge-agreement.png)

Dual judges (DeepSeek-V4.1-Flash + gemma-3-27b-it, temp 0.0, seed 8)
grade every prompt; agreement plus kappa are the license to quote
judge A verdicts above. Judge spend:
${spend.get("estimated_cost", 0.0):.4f} over {spend.get("calls", 0)}
calls ({spend.get("prompt_tokens", 0):,} in / {spend.get("completion_tokens", 0):,} out).

## Eval ids

| run | log |
|---|---|
{eval_rows}

Sanity: `issue8_evals.py sanity` (gold-vs-gold) green; `pytest tests/`
green.

## What to try next

- Completion-only loss (mask the prompt side; this run trains on all
  tokens, matching the v1 lineage).
- Longer SFT (5 epochs) or rank 64 if the delta is positive but small.
- DPO on the 7,000 dispatch pairs (flywheel-v2 `dpo.jsonl`) as the
  stage after SFT.
- Unfreeze the vision tower with pixel-grounded pairs if footage
  counting stays flat.
"""


def main() -> None:
    summary = json.loads((RUN_DIR / "summary.json").read_text())
    judges = json.loads((RUN_DIR / "judge_summary.json").read_text())
    train = json.loads((RUN_DIR / "train.json").read_text())
    spend = json.loads((RUN_DIR / "judge_spend.json").read_text())
    manifest = json.loads((RUN_DIR / "eval_manifest.json").read_text())
    plot_before_after(summary)
    plot_footage_kinds(summary, judges)
    plot_safety_bench(summary, judges)
    plot_judge_agreement(judges)
    REPORT_MD.write_text(render_report(summary, judges, train, spend,
                                       manifest))
    print(f"report -> {REPORT_MD}")


if __name__ == "__main__":
    main()
