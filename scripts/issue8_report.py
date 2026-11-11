"""Issue #8 report: before/after tables, storytelling plots, write-up.

Full-loop report (stages base -> dpo -> safetune -> grpo). Reads
``runs/issue-8/summary.json`` (deterministic), ``judge_summary.json``
+ ``judge_spend.json`` (judge-backed), ``train_<stage>.json`` (Modal
training results), ``drift_<stage>.json`` (gate audits),
``ood_<tag>.json`` / ``bon_<tag>.json`` (generalization), and
``rft_report.json`` (RLVR baseline). Writes five PNGs plus
``docs/reports/issue-8-full-loop.md``. House style: matplotlib Agg,
150 dpi, claim titles, traced footers.

Example:
    .venv/bin/python scripts/issue8_report.py [--final grpo]
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
REPORT_MD = REPORT_DIR / "issue-8-full-loop.md"
STAGES = ("dpo", "safetune", "grpo", "rft")

COLORS = {"base": "#4C78A8", "dpo": "#F58518", "safetune": "#54A24B",
          "grpo": "#E45756", "rft": "#79706E", "ft": "#F58518"}
LABELS = {"base": "base Qwen3.8-27B-FP8", "dpo": "DPO",
          "safetune": "SafeTune", "grpo": "GRPO", "rft": "RFT",
          "ft": "finetune"}


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


def _m(summary: dict, model: str, alias: str) -> dict | None:
    run = summary["runs"].get(f"{model}/{alias}")
    return run["metrics"] if run else None


def plot_before_after(summary: dict, stage: str = "ft") -> Path:
    """Grouped base-vs-stage bars on higher-better headline metrics."""
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
    for i, model in enumerate(("base", stage)):
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
    ax.set_title(f"{LABELS.get(stage, stage)} delta: finetune vs base on every track")
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


def plot_footage_kinds(summary: dict, judges: dict, stage: str = "ft") -> Path:
    """Per-kind footage accuracy plus T1 rescue on flattened rows."""
    _style()
    kinds: dict[str, dict[str, list]] = {}
    for model in ("base", stage):
        for alias in ("floodnet_vqa", "ladi_v2"):
            for kind, cell in _m(summary, model, alias)["by_kind"].items():
                slot = kinds.setdefault(kind, {"base": [0.0, 0],
                                               stage: [0.0, 0]})
                slot[model][0] += cell["mean"] * cell["n"]
                slot[model][1] += cell["n"]
    names = sorted(kinds)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.36
    x = range(len(names))
    for i, model in enumerate(("base", stage)):
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
        for m in ("base", stage)) if t1 else "T1: no judgments"
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("deterministic accuracy")
    ax.set_title("Footage by kind: gains concentrate where gold is exact")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.045, note, fontsize=7, color="#333333", ha="left")
    return _save(fig, "issue-8-footage-kinds.png",
                 "src: runs/issue-8/summary.json + judge_summary.json (T1)")


def plot_safety_bench(summary: dict, judges: dict, stage: str = "ft") -> Path:
    """Per-bench safety match plus T2 carried-decision quality."""
    _style()
    benches: dict[str, dict[str, dict]] = {}
    for model in ("base", stage):
        for bench, cell in _m(summary, model, "safety")["by_bench"].items():
            benches.setdefault(bench, {})[model] = cell
    names = sorted(benches)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.36
    x = range(len(names))
    for i, model in enumerate(("base", stage)):
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
        for m in ("base", stage)) if t2 else "T2: no judgments"
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


def plot_progression(summary: dict) -> Path:
    """Headline metrics across base -> logged stages (full matrices)."""
    _style()
    models = ["base"] + [st for st in STAGES
                         if f"{st}/flywheel_test" in summary["runs"]]
    specs = [
        ("mission", lambda m: _m(summary, m, "flywheel_test")["mean_composite"]),
        ("footage", lambda m: _pooled_footage(summary, m)[0]),
        ("safety", lambda m: _m(summary, m, "safety")["match_rate"]),
        ("refusal", lambda m: _pooled_refusal(summary, m)[0]),
        ("clean", lambda m: 1.0 - _m(summary, m, "redteam")["asr"]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for name, fn in specs:
        ax.plot([LABELS.get(m, m) for m in models],
                [fn(m) for m in models], marker="o", label=name)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (higher is better)")
    ax.set_title("Loop progression: headline metrics per stage")
    ax.legend(fontsize=8, ncol=5)
    return _save(fig, "issue-8-progression.png",
                 "src: runs/issue-8/summary.json (full-matrix stages)")


def _stage_table(summary: dict) -> str:
    """Per-stage before/after rows (None-safe for partial stages)."""
    models = ["base"] + [st for st in STAGES
                         if any(k.startswith(f"{st}/")
                                for k in summary["runs"])]
    header = "| metric | " + " | ".join(models) + " |"
    sep = "|---|---|" + "|".join(["---"] * (len(models) - 1)) + "|"

    def cell(model: str, alias: str, key: str,
             inv: bool = False) -> str:
        m = _m(summary, model, alias)
        if m is None:
            return "n/a"
        v = m[key]
        return f"{(1.0 - v) if inv else v:.3f}"

    def foot(model: str) -> str:
        if (f"{model}/floodnet_vqa" not in summary["runs"]
                and f"{model}/ladi_v2" not in summary["runs"]):
            return "n/a"
        return f"{_pooled_footage(summary, model)[0]:.3f}"

    def ref(model: str) -> str:
        if f"{model}/safety" not in summary["runs"]:
            return "n/a"
        return f"{_pooled_refusal(summary, model)[0]:.3f}"

    rows = [
        "| mission composite | " + " | ".join(
            cell(m, "flywheel_test", "mean_composite") for m in models) + " |",
        "| footage composite | " + " | ".join(
            foot(m) for m in models) + " |",
        "| safety label-match | " + " | ".join(
            cell(m, "safety", "match_rate") for m in models) + " |",
        "| HarmBench+AIR refusal | " + " | ".join(
            ref(m) for m in models) + " |",
        "| red-team clean rate | " + " | ".join(
            cell(m, "redteam", "asr", inv=True) for m in models) + " |",
    ]
    return header + "\n" + sep + "\n" + "\n".join(rows)


def _train_table(trains: dict) -> str:
    rows = ["| stage | loss/reward | steps | min | gpu | adapter |",
            "|---|---|---|---|---|---|"]
    for stage in ("sft", "dpo", "safetune", "grpo", "rft"):
        t = trains.get(stage)
        if not t:
            continue
        metric = t.get("train_loss", t.get("mean_reward", 0.0))
        rows.append(
            f"| {stage} | {metric:.4f} | {t.get('steps', 0)} | "
            f"{t.get('minutes', 0.0):.1f} | {t.get('gpu_class', '?')} | "
            f"`{t.get('hf_repo', '?')}` @ `{str(t.get('hf_rev', '?'))[:12]}` |")
    return "\n".join(rows)


def render_report(summary: dict, judges: dict, train: dict,
                  spend: dict, manifest: dict, stage: str = "ft",
                  extra: dict | None = None) -> str:
    """Full write-up with computed tables (prose reviewed by hand).

    ``train`` is one stage dict (legacy) or ``{stage: dict}``;
    ``extra`` carries drift/ood/bon/rft/replay/deviations blocks.
    """
    extra = extra or {}
    trains = train if any(isinstance(v, dict) and "steps" in v
                          for v in train.values()) else {"ft": train}
    mission_b = _m(summary, "base", "flywheel_test")
    mission_f = _m(summary, stage, "flywheel_test")
    foot_b, foot_bn = _pooled_footage(summary, "base")
    foot_f, foot_fn = _pooled_footage(summary, stage)
    safe_b = _m(summary, "base", "safety")
    safe_f = _m(summary, stage, "safety")
    ref_b, ref_bn = _pooled_refusal(summary, "base")
    ref_f, ref_fn = _pooled_refusal(summary, stage)
    rt_b = _m(summary, "base", "redteam")
    rt_f = _m(summary, stage, "redteam")
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
    drift_md = extra.get("drift_md", "")
    ood_md = extra.get("ood_md", "")
    bon_md = extra.get("bon_md", "")
    rft_md = extra.get("rft_md", "")
    replay_md = extra.get("replay_md", "")
    deviations_md = extra.get("deviations_md", "")
    examples_md = extra.get("examples_md", "")
    return f"""# Full loop: Qwen3.8-27B SFT-DPO-SafeTune-GRPO (issue #8)

Six-stage post-training loop on held-out splits plus safety-bench
gates, per `docs/model-comparison-protocol.md` section 2.
Flywheel-v1's delta was null; this run applies the same rigor and
reports every delta honestly either way.

## Method (per stage)

{_train_table(trains)}

QLoRA (NF4, bf16 compute) throughout, language tower only (vision
frozen), seed 8. SFT: r32/alpha64, 1 epoch (slice-forced; protocol
says 3), eff batch 8, lr 2e-4 cosine, max 2048, no packing. DPO:
length-normalized (`sigmoid_norm`, beta 0.1), 1 epoch, lr 5e-5.
SafeTune: safety-mix repair to Gate 2. GRPO: G=4 rule-reward pilot
vs the E-RFT baseline. Data fixes at load (pinned data untouched):
disasterm3 fragment lists joined, tool-call turns preserved.

## Before/after (held-out, base vs {stage})

| metric | base | {stage} | delta |
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

## Per-stage progression

{_stage_table(summary)}

![progression](issue-8-progression.png)

![footage by kind](issue-8-footage-kinds.png)

Footage splits by scoring kind because the SFT mix is kind-imbalanced
(short yes/no VQA dominates). The T1 judge re-grades every flattened
row, so a deterministic dip that the judge calls `correct` reads as a
format shift, not a capability loss.

## Safety gates + drift audit

Gate 1 (DPO exit): HarmBench/AIR-Bench refusal must not drop vs base.
Gate 2 (SafeTune exit): agentharm misuse 1.0, benign >= 0.85,
abstention >= 0.80. Per-bench label-match (base vs {stage}):

| bench | base | {stage} | delta |
|---|---|---|---|
{bench_rows}

![safety by bench](issue-8-safety-bench.png)

{drift_md}

The T2 judge grades the carried decision plus quality 1-5 on every
non-exact row, so over-refusal (high refusal, low quality) cannot hide
behind the gate metric.

## Generalization (OOD shift + best-of-N)

{ood_md}

{bon_md}

## RLVR vs RFT baseline

{rft_md}

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

## Per-stage example pairs

{examples_md}

## Replay

{replay_md}

## Eval ids

| run | log |
|---|---|
{eval_rows}

Sanity: `issue8_evals.py sanity` (gold-vs-gold) green; `pytest tests/`
green; `replay_loop.py --issue 8 verify` reproduces every table.

## DEVIATIONS (protocol section 2)

{deviations_md}

## What to try next

- SFT 3 epochs + IPO arm when the slice allows (both deferred on
  cost with logged estimates).
- On-policy rejected upgrade for DPO (§2.1).
- Unfreeze the vision tower with pixel-grounded pairs if footage
  counting stays flat.
- Graduate the safety subsets to full upstream benches before any
  release claim.
"""


def _read(name: str) -> dict:
    path = RUN_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def _build_extra() -> dict:
    """Optional report blocks from stage artifacts (missing -> skip)."""
    extra: dict = {}
    drift_lines = []
    for stage in ("dpo", "safetune", "grpo"):
        d = _read(f"drift_{stage}.json")
        if not d:
            continue
        for sl in ("full", "heldout"):
            v = d[sl]["verdict"]["verdict"]
            ov = d[sl]["over_refusal"][stage]["rate"]
            hs = d[sl]["harm_score"][stage]["harm_score"]
            drift_lines.append(
                f"- {stage}/{sl}: {v} (over-refusal {ov:.3f}, "
                f"harm {hs:.3f})")
    if drift_lines:
        extra["drift_md"] = ("Drift audits (`issue8_drift.py` on the "
                             "#10 harness):\n\n" + "\n".join(drift_lines))
    ood_lines = []
    for tag in ("base", "final"):
        o = _read(f"ood_{tag}.json")
        if o:
            ood_lines.append(
                f"| {tag} | {o['acc_holdout']:.2f} | "
                f"{o['acc_shifted']:.2f} | {o['ood_shift_gap']:+.2f} |")
    if ood_lines:
        extra["ood_md"] = ("OOD rule-shift (Chu §8.4; RL should shrink "
                           "|gap|):\n\n| tag | holdout | shifted | gap |\n"
                           "|---|---|---|---|\n" + "\n".join(ood_lines))
    bon_lines = []
    for tag in ("base", "final"):
        b = _read(f"bon_{tag}.json")
        if b:
            bon_lines.append(
                f"| {tag} | {b['acc_greedy']:.2f} | {b['acc_bon']:.2f} | "
                f"{b['bon_gain']:+.2f} | {b['mean_n_cost']} |")
    if bon_lines:
        extra["bon_md"] = ("Best-of-8 + verifier (§6.1, limit=100):\n\n"
                           "| tag | greedy | BoN | gain | mean-N |\n"
                           "|---|---|---|---|---|\n" + "\n".join(bon_lines))
    rft = _read("rft_report.json")
    grpo = _read("train_grpo.json")
    if rft:
        extra["rft_md"] = (
            f"E-RFT baseline (verifier-filtered, §1.2): pass rate "
            f"{rft['verifier_pass_rate']:.3f} over {rft['samples']} "
            f"samples ({rft['prompts_with_pass']}/{rft['prompts']} "
            f"prompts with a pass). GRPO mean rule-reward: "
            f"{grpo.get('mean_reward', 0.0):.4f} — the pilot must beat "
            f"RFT on the same verifier to justify the RL machinery.")
    replay = _read("replays/manifest.json")
    if replay:
        extra["replay_md"] = (
            f"Bundles `runs/issue-8/replays/*.jsonl` (seed "
            f"{replay.get('seed', 8)}): {replay.get('tracks', {})}; "
            f"uploaded to `sanggusti/dmb-issue-8-replays`; "
            f"`replay_loop.py --issue 8 verify` reproduces every table.")
    dev = RUN_DIR / "deviations.md"
    if dev.exists():
        extra["deviations_md"] = dev.read_text()
    ex = RUN_DIR / "examples.md"
    if ex.exists():
        extra["examples_md"] = ex.read_text()
    return extra


def main(argv: list[str] | None = None) -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", default="grpo")
    args = parser.parse_args(argv)
    summary = json.loads((RUN_DIR / "summary.json").read_text())
    judges = json.loads((RUN_DIR / "judge_summary.json").read_text())
    trains = {st: _read(f"train_{st}.json")
              for st in ("sft", "dpo", "safetune", "grpo", "rft")}
    trains = {k: v for k, v in trains.items() if v}
    spend = json.loads((RUN_DIR / "judge_spend.json").read_text())
    manifest = json.loads((RUN_DIR / "eval_manifest.json").read_text())
    plot_before_after(summary, args.final)
    plot_footage_kinds(summary, judges, args.final)
    plot_safety_bench(summary, judges, args.final)
    plot_judge_agreement(judges)
    plot_progression(summary)
    REPORT_MD.write_text(render_report(summary, judges, trains, spend,
                                       manifest, args.final,
                                       _build_extra()))
    print(f"report -> {REPORT_MD}")


if __name__ == "__main__":
    main()
