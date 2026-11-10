"""Issue-74 eval loop: prompts, deterministic scores, judges, plots.

Pipeline (run from the repo root):
    python scripts/aya74_eval.py prompts     # build runs/aya74/eval_prompts.jsonl
    # ... Modal generate for base/sft/dpo, download to runs/aya74/ ...
    python scripts/aya74_eval.py score       # deterministic lanes -> det_scores.json
    python scripts/aya74_eval.py judge-estimate
    python scripts/aya74_eval.py judge-run   # HF router judges (see below)
    python scripts/aya74_eval.py analyze     # agreement + CIs -> analysis.json
    python scripts/aya74_eval.py plot        # docs/reports/issue-74-*.png

Judges follow docs/judge-playbook.md triggers (T1-T5, reasoning+verdict).
Issue #74 allows Meta muse-wonder-1.0 or the HF router; this environment
carries no Meta API key, so judges run on the HF router with the same
dual-judge pair as the #62 pilot (DeepSeek-V4.1-Flash + gemma-3-27b-it).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.judge.templates import (  # noqa: E402
    ABSTENTION_LABELS,
    ABSTENTION_SYSTEM,
    DISPATCH_SYSTEM,
    FOOTAGE_SYSTEM,
    REDTEAM_LABELS,
    REDTEAM_SYSTEM,
    SAFETY_LABELS,
    SAFETY_SYSTEM,
    estimate_tokens,
    parse_judgment,
    parse_pairwise,
    render_abstention,
    render_dispatch_pair,
    render_footage,
    render_redteam,
    render_safety,
)
from hsea.scorers.abstention import compute_abstention_scores  # noqa: E402
from hsea.scorers.footage import compute_footage  # noqa: E402
from hsea.scorers.redteam import compute_redteam_scores  # noqa: E402
from hsea.scorers.safety import compute_safety  # noqa: E402

SEED = 74
RUN_DIR = ROOT / "runs" / "aya74"
REPORT_DIR = ROOT / "docs" / "reports"
CHECKPOINTS = ("base", "sft", "dpo")

JUDGE_A = "deepseek-ai/DeepSeek-V4.1-Flash"
JUDGE_B = "google/gemma-3-27b-it"
HF_BASE_URL = "https://router.huggingface.co/v1"
RATE_IN = 0.30
RATE_OUT = 1.20
SPEND_CAP = 5.00
OUT_TOKENS_BOUND = 600


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def cmd_prompts() -> None:
    """Build the eval prompt file (exact prior-eval systems + inputs)."""
    from evals.abstention import ABSTENTION_SYSTEM
    from evals.redteam import REDTEAM_SYSTEM
    from hsea.data.abstention import load_scenarios
    from hsea.data.footage import (FOOTAGE_SYSTEM, build_floodnet_samples,
                                   build_ladi_samples)
    from hsea.data.redteam import load_attacks
    from hsea.safety.data import ITEMS, SAFETY_SYSTEM

    prompts: list[dict] = []
    for idx, (bench, category, label, prompt) in enumerate(ITEMS):
        prompts.append({"key": f"safety:{bench}:{idx}", "track": "safety",
                        "system": SAFETY_SYSTEM, "prompt": prompt,
                        "gold": label,
                        "meta": {"bench": bench, "category": category,
                                 "label": label}})
    for rec in load_scenarios():
        prompts.append({"key": f"abstention:{rec['id']}",
                        "track": "abstention", "system": ABSTENTION_SYSTEM,
                        "prompt": rec["prompt"], "gold": rec["label"],
                        "meta": {"label": rec["label"], "set": rec["set"],
                                 "split": rec.get("split", ""),
                                 "stakes": rec.get("stakes", ""),
                                 "category": rec.get("category", "")}})
    for rec in load_attacks():
        prompts.append({"key": f"redteam:{rec['id']}", "track": "redteam",
                        "system": REDTEAM_SYSTEM, "prompt": rec["prompt"],
                        "gold": "refuse",
                        "meta": {"family": rec["family"],
                                 "variant": rec["variant"],
                                 "severity": rec["severity"],
                                 "request": rec["request"],
                                 "markers": list(rec.get("markers", []))}})
    try:
        flood = build_floodnet_samples()
    except Exception as exc:  # noqa: BLE001 - footage is best-effort text floor
        print(f"floodnet skipped: {type(exc).__name__}: {exc}")
        flood = []
    for s in flood:
        meta = dict(s.metadata or {})
        prompts.append({"key": f"footage:floodnet:{s.id}", "track": "footage",
                        "system": FOOTAGE_SYSTEM, "prompt": _text(s.input),
                        "gold": str(meta.get("answer", "")),
                        "meta": {"kind": meta.get("kind", "vqa"),
                                 "answer": str(meta.get("answer", "")),
                                 "corpus": "floodnet"}})
    try:
        ladi = build_ladi_samples()[:24]
    except Exception as exc:  # noqa: BLE001 - footage is best-effort text floor
        print(f"ladi skipped: {type(exc).__name__}: {exc}")
        ladi = []
    for s in ladi:
        meta = dict(s.metadata or {})
        prompts.append({"key": f"footage:ladi:{s.id}", "track": "footage",
                        "system": FOOTAGE_SYSTEM, "prompt": _text(s.input),
                        "gold": str(meta.get("answer", "")),
                        "meta": {"kind": meta.get("kind", "vqa"),
                                 "answer": str(meta.get("answer", "")),
                                 "corpus": "ladi"}})
    for line in open(ROOT / "data" / "dispatch-pairs" / "dpo.jsonl"):
        pair = json.loads(line)
        pid = pair["provenance"]["record_id"]
        prompts.append({"key": f"dispatch:{pid}", "track": "dispatch",
                        "system": "", "prompt": f"Situation: {pair['situation']}\n"
                        f"Decide the dispatch priority and action.",
                        "gold": "", "meta": {"priority": pair["priority"]}})

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "eval_prompts.jsonl", "w") as fh:
        for rec in prompts:
            fh.write(json.dumps(rec) + "\n")
    from collections import Counter
    print(f"prompts={len(prompts)} {dict(Counter(p['track'] for p in prompts))}"
          f" -> {RUN_DIR / 'eval_prompts.jsonl'}")


def _load_completions() -> tuple[list[dict], dict[str, dict[str, str]]]:
    prompts = [json.loads(line)
               for line in open(RUN_DIR / "eval_prompts.jsonl")]
    completions: dict[str, dict[str, str]] = {}
    for ckpt in CHECKPOINTS:
        path = RUN_DIR / f"eval_{ckpt}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} missing, skipping {ckpt}")
            continue
        completions[ckpt] = {}
        for line in open(path):
            rec = json.loads(line)
            completions[ckpt][rec["key"]] = rec.get("completion", "")
    return prompts, completions


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a rate (total, edge-safe)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cmd_score() -> None:
    """Deterministic lanes per checkpoint -> det_scores.json + tables."""
    prompts, completions = _load_completions()
    assert completions, "no completion files; run Modal generate first"
    by_track: dict[str, list[dict]] = {}
    for rec in prompts:
        by_track.setdefault(rec["track"], []).append(rec)

    summary: dict = {"checkpoints": {}, "n_prompts": len(prompts)}
    for ckpt, comps in completions.items():
        rows: list[dict] = []
        for rec in prompts:
            comp = comps.get(rec["key"], "")
            rows.append({"key": rec["key"], "track": rec["track"],
                         "completion": comp})
        cell: dict = {}

        s_rows = [r for r in rows if r["track"] == "safety"]
        s_vecs = [compute_safety(
            r["completion"],
            next(p for p in prompts if p["key"] == r["key"])["gold"])
            for r in s_rows]
        ok = sum(1 for v in s_vecs if v["safety_correct"] == 1.0)
        by_label: dict[str, dict] = {}
        for r, v in zip(s_rows, s_vecs):
            gold = next(p for p in prompts if p["key"] == r["key"])["gold"]
            d = by_label.setdefault(gold, {"n": 0, "ok": 0})
            d["n"] += 1
            d["ok"] += 1 if v["safety_correct"] == 1.0 else 0
        cell["safety"] = {"n": len(s_rows), "exact": ok,
                          "rate": ok / len(s_rows) if s_rows else None,
                          "ci": wilson(ok, len(s_rows)), "by_label": by_label}

        a_rows = [r for r in rows if r["track"] == "abstention"]
        a_vecs = [compute_abstention_scores(
            r["completion"],
            next(p for p in prompts if p["key"] == r["key"])["gold"])
            for r in a_rows]
        a_ok = sum(1 for v in a_vecs if v["abst_exact"] == 1.0)
        a_soft = sum(float(v["abst_soft"]) for v in a_vecs)
        a_footer = sum(1 for v in a_vecs if v["abst_pred"] != "none")
        cell["abstention"] = {"n": len(a_rows), "exact": a_ok,
                              "rate": a_ok / len(a_rows) if a_rows else None,
                              "ci": wilson(a_ok, len(a_rows)),
                              "soft_mean": a_soft / len(a_vecs) if a_vecs else None,
                              "footer_rate": a_footer / len(a_vecs) if a_vecs else None}

        r_rows = [r for r in rows if r["track"] == "redteam"]
        r_vecs = []
        for r in r_rows:
            meta = next(p for p in prompts if p["key"] == r["key"])["meta"]
            r_vecs.append(compute_redteam_scores(
                r["completion"], meta["family"], meta["severity"],
                meta.get("markers", [])))
        asr_n = sum(float(v["asr"]) for v in r_vecs)
        review = sum(1 for v in r_vecs if v["rt_needs_review"] == 1.0)
        by_family: dict[str, dict] = {}
        for r, v in zip(r_rows, r_vecs):
            fam = next(p for p in prompts if p["key"] == r["key"])["meta"]["family"]
            d = by_family.setdefault(fam, {"n": 0, "asr": 0.0})
            d["n"] += 1
            d["asr"] += float(v["asr"])
        cell["redteam"] = {"n": len(r_rows), "asr_sum": asr_n,
                           "asr": asr_n / len(r_vecs) if r_vecs else None,
                           "needs_review": review, "by_family": by_family}

        f_rows = [r for r in rows if r["track"] == "footage"]
        f_vecs = []
        for r in f_rows:
            meta = next(p for p in prompts if p["key"] == r["key"])["meta"]
            f_vecs.append(compute_footage(r["completion"], meta))
        f_ok = sum(1 for v in f_vecs if v["footage_composite"] == 1.0)
        by_kind: dict[str, dict] = {}
        for r, v in zip(f_rows, f_vecs):
            kind = next(p for p in prompts if p["key"] == r["key"])["meta"]["kind"]
            d = by_kind.setdefault(kind, {"n": 0, "ok": 0})
            d["n"] += 1
            d["ok"] += 1 if v["footage_composite"] == 1.0 else 0
        cell["footage"] = {"n": len(f_rows), "exact": f_ok,
                           "rate": f_ok / len(f_rows) if f_rows else None,
                           "ci": wilson(f_ok, len(f_rows)), "by_kind": by_kind}
        cell["dispatch_n"] = sum(1 for r in rows if r["track"] == "dispatch")
        summary["checkpoints"][ckpt] = cell

    with open(RUN_DIR / "det_scores.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("track      | " + " | ".join(f"{c:>16}" for c in summary["checkpoints"]))
    for track, fmt in (("safety", "rate"), ("abstention", "rate"),
                       ("redteam", "asr"), ("footage", "rate")):
        line = f"{track:10} | "
        for ckpt in summary["checkpoints"]:
            cell = summary["checkpoints"][ckpt][track]
            val = cell[fmt]
            n = cell["n"]
            line += f"{val if val is None else f'{val:.3f}'} (n={n}) | "
        print(line)


def _split(full: str, system: str) -> tuple[str, str]:
    assert full.startswith(system), "render must start with system"
    return system, full[len(system):].strip()


def _judge_plan() -> list[tuple[str, str, str, str]]:
    """(key, judge-kind, system, user) per the playbook triggers.

    Primary checkpoint is dpo; every triggered key is paired with the
    base completion on the same key so deltas are paired, not pooled.
    """
    prompts, completions = _load_completions()
    assert "dpo" in completions and "base" in completions, \
        "need base + dpo completions for paired judging"
    by_key = {p["key"]: p for p in prompts}
    rng = random.Random(SEED)
    out: list[tuple[str, str, str, str]] = []

    def add(kind: str, key: str, ckpt: str, full: str, system: str):
        s, u = _split(full, system)
        out.append((f"{kind}:{ckpt}:{key}", kind, s, u))

    safety = [p for p in prompts if p["track"] == "safety"]
    bad, good = [], []
    for p in safety:
        v = compute_safety(completions["dpo"].get(p["key"], ""), p["gold"])
        (bad if v["safety_correct"] != 1.0 else good).append(p)
    rng.shuffle(good)
    audit = good[:max(1, len(good) // 10)]
    for p in bad + audit:
        for ckpt in ("dpo", "base"):
            full = render_safety(p["prompt"], p["gold"],
                                 completions[ckpt].get(p["key"], ""), seed=SEED)
            add("T2", p["key"], ckpt, full, SAFETY_SYSTEM)

    abst = [p for p in prompts if p["track"] == "abstention"]
    trig, rest = [], []
    for p in abst:
        v = compute_abstention_scores(completions["dpo"].get(p["key"], ""),
                                      p["gold"])
        if v["abst_pred"] == "none" or v["abst_soft"] == 0.5:
            trig.append(p)
        else:
            rest.append(p)
    rng.shuffle(rest)
    for p in trig + rest[:max(1, len(rest) // 10)]:
        for ckpt in ("dpo", "base"):
            full = render_abstention(
                p["prompt"], p["meta"].get("stakes", ""), p["gold"],
                completions[ckpt].get(p["key"], ""), seed=SEED)
            add("T3", p["key"], ckpt, full, ABSTENTION_SYSTEM)

    red = [p for p in prompts if p["track"] == "redteam"]
    review, asr1, rest = [], [], []
    for p in red:
        v = compute_redteam_scores(completions["dpo"].get(p["key"], ""),
                                   p["meta"]["family"], p["meta"]["severity"],
                                   p["meta"].get("markers", []))
        if v["rt_needs_review"] == 1.0:
            review.append(p)
        elif v["asr"] == 1.0:
            asr1.append(p)
        else:
            rest.append(p)
    rng.shuffle(asr1)
    for p in review + asr1[:8]:
        for ckpt in ("dpo", "base"):
            full = render_redteam(
                p["meta"]["family"], p["meta"]["variant"],
                p["meta"]["severity"], p["meta"]["request"],
                p["meta"].get("markers", []),
                completions[ckpt].get(p["key"], ""), seed=SEED)
            add("T4", p["key"], ckpt, full, REDTEAM_SYSTEM)

    foot = [p for p in prompts if p["track"] == "footage"]
    zero, one = [], []
    for p in foot:
        v = compute_footage(completions["dpo"].get(p["key"], ""), p["meta"])
        comp = completions["dpo"].get(p["key"], "")
        if v["footage_composite"] != 1.0 and comp.strip():
            zero.append(p)
        elif v["footage_composite"] == 1.0:
            one.append(p)
    rng.shuffle(one)
    for p in zero + one[:max(1, len(one) // 10)]:
        for ckpt in ("dpo", "base"):
            full = render_footage(p["meta"]["kind"], p["prompt"], p["gold"],
                                  completions[ckpt].get(p["key"], ""), seed=SEED)
            add("T1", p["key"], ckpt, full, FOOTAGE_SYSTEM)

    dispatch = [p for p in prompts if p["track"] == "dispatch"]
    for p in dispatch:
        situation = p["prompt"].split("\nDecide")[0].replace("Situation: ", "")
        base_c = completions["base"].get(p["key"], "")
        dpo_c = completions["dpo"].get(p["key"], "")
        for order, (a, b) in (("orig", (base_c, dpo_c)), ("swap", (dpo_c, base_c))):
            full = render_dispatch_pair(situation, a, b, seed=SEED)
            s, u = _split(full, DISPATCH_SYSTEM)
            out.append((f"T5:{order}:{p['key']}", "T5", s, u))
    print(f"judge plan: {len(out)} prompts "
          f"({len([o for o in out if o[1] == 'T5'])} T5)")
    void = by_key  # keys trace to prompts; kept for debuggability
    assert void
    return out


def cmd_judge_estimate() -> None:
    """Upper-bound judge cost; aborts when over the cap."""
    plan = _judge_plan()
    calls = len(plan) * 2  # dual judges
    in_tokens = sum(estimate_tokens(s) + estimate_tokens(u)
                    for _, _, s, u in plan) * 2
    out_tokens = calls * OUT_TOKENS_BOUND
    upper = (in_tokens * RATE_IN + out_tokens * RATE_OUT) / 1_000_000
    print(f"prompts={len(plan)} calls={calls} "
          f"in~{in_tokens} out<={out_tokens}")
    print(f"upper-bound=${upper:.4f} cap=${SPEND_CAP:.2f}")
    if upper > SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _judge_client():
    from openai import OpenAI

    key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_TOKEN")
    if not key:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            key = token_file.read_text().strip()
    if not key:
        raise SystemExit("missing HF_API_KEY (or HUGGINGFACE_TOKEN)")
    return OpenAI(api_key=key, base_url=HF_BASE_URL)


def cmd_judge_run() -> None:
    """Call both judges over the plan; ledger every call."""
    plan = _judge_plan()
    client = _judge_client()
    t0 = time.time()
    rows: list[dict] = []
    for key, kind, system, user in plan:
        for judge in (JUDGE_A, JUDGE_B):
            resp = client.chat.completions.create(
                model=judge, temperature=0.0, max_tokens=OUT_TOKENS_BOUND,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            msg = resp.choices[0].message.content or ""
            u = resp.usage
            try:
                cost = float(getattr(u, "estimated_cost", None) or 0.0)
            except (TypeError, ValueError):
                cost = None
            rows.append({
                "key": key, "kind": kind, "judge": judge, "text": msg,
                "prompt_tokens": int(u.prompt_tokens or 0),
                "completion_tokens": int(u.completion_tokens or 0),
                "estimated_cost": cost,
            })
            print(f"{key} {judge.split('/')[-1]} "
                  f"in={u.prompt_tokens} out={u.completion_tokens} "
                  f"${(cost or 0.0):.6f}", flush=True)
    minutes = (time.time() - t0) / 60
    total_cost = sum(r["estimated_cost"] or 0.0 for r in rows)
    with open(RUN_DIR / "judgments.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"calls": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "estimated_cost": total_cost, "minutes": minutes}
    with open(RUN_DIR / "judge_spend.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


def cmd_analyze() -> None:
    """Judge-vs-deterministic deltas + T5 win rates -> analysis.json."""
    prompts = {json.loads(line)["key"]: json.loads(line)
               for line in open(RUN_DIR / "eval_prompts.jsonl")}
    judgments: dict[str, dict[str, dict]] = {}
    for line in open(RUN_DIR / "judgments.jsonl"):
        row = json.loads(line)
        judgments.setdefault(row["key"], {})[row["judge"]] = row

    summary: dict = {"single": {}, "t5": {}}
    for kind, labels in (("T2", SAFETY_LABELS), ("T3", ABSTENTION_LABELS),
                         ("T4", REDTEAM_LABELS), ("T1", ("correct", "close", "wrong"))):
        keys = sorted({k for k in judgments if k.startswith(kind + ":dpo:")})
        base_keys = sorted({k for k in judgments if k.startswith(kind + ":base:")})
        parsed_ok = sum(1 for k in keys
                        if parse_judgment(judgments[k][JUDGE_A]["text"], labels)["ok"])
        base_ok = sum(1 for k in base_keys
                      if parse_judgment(judgments[k][JUDGE_A]["text"], labels)["ok"])
        summary["single"][kind] = {"dpo_keys": len(keys), "base_keys": len(base_keys),
                                   "dpo_parse": parsed_ok, "base_parse": base_ok,
                                   "prompts_traced": all(
                                       k.split(":", 2)[2] in prompts for k in keys)}

    t5_keys = sorted({k for k in judgments if k.startswith("T5:")})
    wins = {"dpo": 0, "base": 0, "tie": 0, "inconsistent": 0}
    pairs: dict[str, dict[str, dict]] = {}
    for k in t5_keys:
        _, order, pid = k.split(":", 2)
        pairs.setdefault(pid, {})[order] = parse_pairwise(
            judgments[k][JUDGE_A]["text"])
    for pid, orders in pairs.items():
        o, s = orders.get("orig", {}).get("verdict"), orders.get("swap", {}).get("verdict")
        flip = {"A": "B", "B": "A", "C": "C"}
        if o not in flip or s != flip[o]:
            wins["inconsistent"] += 1
        elif o == "C":
            wins["tie"] += 1
        elif o == "B":  # orig order is (base=A, dpo=B)
            wins["dpo"] += 1
        else:
            wins["base"] += 1
    n = len(pairs)
    summary["t5"] = {"pairs": n, **wins,
                     "dpo_win_rate": wins["dpo"] / n if n else None,
                     "dpo_win_ci": wilson(wins["dpo"], n)}
    with open(RUN_DIR / "analysis.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def cmd_plot() -> None:
    """Storytelling plots for the report (each argued in prose there)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det = json.load(open(RUN_DIR / "det_scores.json"))
    ckpts = [c for c in CHECKPOINTS if c in det["checkpoints"]]
    analysis = json.load(open(RUN_DIR / "analysis.json")) if \
        (RUN_DIR / "analysis.json").exists() else {}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    tracks = [("safety", "rate", "safety exact"),
              ("abstention", "rate", "abstention exact"),
              ("redteam", "asr", "red-team ASR (lower better)"),
              ("footage", "rate", "footage text-floor exact")]
    x = range(len(tracks))
    width = 0.22
    for i, ckpt in enumerate(ckpts):
        vals, lo, hi = [], [], []
        for track, field, _ in tracks:
            cell = det["checkpoints"][ckpt][track]
            v = cell[field]
            vals.append(v if v is not None else 0.0)
            if track == "redteam":
                lo.append(0.0)
                hi.append(0.0)
            else:
                ci_lo, ci_hi = cell["ci"]
                lo.append(vals[-1] - ci_lo)
                hi.append(ci_hi - vals[-1])
        pos = [j + (i - 1) * width for j in x]
        ax.bar(pos, vals, width, yerr=[lo, hi], capsize=3, label=ckpt)
    ax.set_xticks(list(x), [label for _, _, label in tracks], rotation=12,
                  ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate (Wilson 95% where shown)")
    ax.set_title("Deterministic lanes: base vs SFT vs DPO (issue #74)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-74-det-lanes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for ckpt in ckpts:
        by_label = det["checkpoints"][ckpt]["safety"]["by_label"]
        comp = by_label.get("comply", {"n": 0, "ok": 0})
        ref = by_label.get("refuse", {"n": 0, "ok": 0})
        x_val = comp["ok"] / comp["n"] if comp["n"] else 0.0
        y_val = ref["ok"] / ref["n"] if ref["n"] else 0.0
        ax.scatter([x_val], [y_val], s=90, label=f"{ckpt} (n={comp['n'] + ref['n']})")
        ax.annotate(ckpt, (x_val, y_val), textcoords="offset points",
                    xytext=(6, 6), fontsize=10)
    ax.set_xlabel("benign comply rate (utility)")
    ax.set_ylabel("refuse rate on harmful rows (safety)")
    ax.set_title("Safety Pareto: refusal vs benign utility per checkpoint")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-74-safety-pareto.png", dpi=150)
    plt.close(fig)

    t5 = (analysis.get("t5") or {})
    if t5.get("pairs"):
        fig, ax = plt.subplots(figsize=(7, 4.4))
        labels = ["dpo wins", "base wins", "tie", "inconsistent"]
        vals = [t5.get("dpo", 0), t5.get("base", 0), t5.get("tie", 0),
                t5.get("inconsistent", 0)]
        ax.bar(labels, vals)
        for label, val in zip(labels, vals):
            ax.text(label, val + 0.1, str(val), ha="center", fontsize=11)
        ax.set_ylabel("dispatch pairs (n=%d, swap-twice)" % t5["pairs"])
        ax.set_title("T5 dispatch duels: DPO vs base (judge A)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "issue-74-t5-duels.png", dpi=150)
        plt.close(fig)

    log_path = RUN_DIR / "training_curves.json"
    if log_path.exists():
        curves = json.load(open(log_path))
        fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4.2),
                                 squeeze=False)
        for ax, (stage, points) in zip(axes[0], curves.items()):
            steps = [p[0] for p in points]
            losses = [p[1] for p in points]
            ax.plot(steps, losses, marker="o", markersize=3)
            ax.set_xlabel("logging step")
            ax.set_ylabel("loss")
            ax.set_title(f"{stage} training loss")
        fig.suptitle("Training loss curves (Modal trainer log history)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "issue-74-loss-curves.png", dpi=150)
        plt.close(fig)
    print(f"plots -> {REPORT_DIR}/issue-74-*.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="issue-74 eval loop")
    parser.add_argument("command", choices=["prompts", "score", "judge-estimate",
                                            "judge-run", "analyze", "plot"])
    args = parser.parse_args(argv)
    {"prompts": cmd_prompts, "score": cmd_score,
     "judge-estimate": cmd_judge_estimate, "judge-run": cmd_judge_run,
     "analyze": cmd_analyze, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
