"""Issue-74 Stage 6: Meta ember-minimal third judge + 3-way agreement.

Reuses the Phase-1 judge plan (``aya74_eval._judge_plan``, 174 prompts
over the recorded base/DPO completions) and adds Meta ember-minimal
(the #72 calibrated default) alongside the committed HF dual pair.
No re-spend on HF: ``judgments.jsonl`` is read, never rewritten.

Commands (run from the repo root; key from META_AI_API_KEY):
    python scripts/aya74_meta_judge.py estimate
    python scripts/aya74_meta_judge.py run       # -> judgments_meta.jsonl
    python scripts/aya74_meta_judge.py analyze   # -> agreement3.json
    python scripts/aya74_meta_judge.py plot      # 3-way agreement bars
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aya74_eval import _judge_plan, wilson  # noqa: E402
from hsea.judge.providers import (  # noqa: E402
    META_BASE_URL,
    META_JUDGE_CONFIGS,
    META_RATES,
    judgment_cost,
)
from hsea.judge.templates import (  # noqa: E402
    ABSTENTION_LABELS,
    REDTEAM_LABELS,
    SAFETY_LABELS,
    estimate_tokens,
    parse_judgment,
    parse_pairwise,
)
from hsea.stats import cohen_kappa  # noqa: E402

RUN_DIR = ROOT / "runs" / "aya74"
REPORT_DIR = ROOT / "docs" / "reports"
CONFIG = "ember-minimal"
SPEND_CAP = 2.00
MAX_RETRIES = 3

TRACK_LABELS = {
    "T2": SAFETY_LABELS,
    "T3": ABSTENTION_LABELS,
    "T4": REDTEAM_LABELS,
    "T1": ("correct", "close", "wrong"),
}


def _key() -> str:
    key = os.environ.get("META_AI_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in open(ROOT / ".env"):
            if line.startswith("META_AI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("missing META_AI_API_KEY (env or .env)")
    return key


def cmd_estimate() -> None:
    plan = _judge_plan()
    cfg = META_JUDGE_CONFIGS[CONFIG]
    rates = META_RATES[str(cfg["model"])]
    in_tokens = sum(estimate_tokens(s) + estimate_tokens(u)
                    for _, _, s, u in plan)
    upper = (in_tokens * rates["input"]
             + len(plan) * int(cfg["max_tokens"]) * rates["output"]) / 1e6
    print(f"config={CONFIG} prompts={len(plan)} in~{in_tokens} "
          f"upper=${upper:.4f} cap=${SPEND_CAP:.2f}")
    if upper > SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _call(client, model: str, effort: str, max_out: int,
          system: str, user: str) -> dict:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=max_out,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                reasoning_effort=effort)
            dt = time.time() - t0
            u = resp.usage
            det = u.completion_tokens_details
            return {
                "text": resp.choices[0].message.content or "",
                "finish": resp.choices[0].finish_reason,
                "prompt_tokens": int(u.prompt_tokens or 0),
                "completion_tokens": int(u.completion_tokens or 0),
                "reasoning_tokens": int(
                    getattr(det, "reasoning_tokens", 0) or 0),
                "latency_s": round(dt, 1),
            }
        except Exception as exc:  # noqa: BLE001, retry then record
            last = exc
            time.sleep(2.0 * (attempt + 1))
    return {"error": f"{type(last).__name__}: {str(last)[:200]}"}


def cmd_run() -> None:
    from openai import OpenAI

    plan = _judge_plan()
    cfg = META_JUDGE_CONFIGS[CONFIG]
    model = str(cfg["model"])
    effort = str(cfg["reasoning_effort"])
    max_out = int(cfg["max_tokens"])
    client = OpenAI(api_key=_key(), base_url=META_BASE_URL)
    t0 = time.time()
    rows: list[dict] = []
    for key, kind, system, user in plan:
        res = _call(client, model, effort, max_out, system, user)
        if "error" in res:
            rows.append({"key": key, "kind": kind, "judge": "meta-ember",
                         "config": CONFIG, "model": model,
                         "error": res["error"]})
            print(f"{key} ERROR {res['error']}", flush=True)
            continue
        cost = judgment_cost(res["prompt_tokens"],
                             res["completion_tokens"], model)
        rows.append({"key": key, "kind": kind, "judge": "meta-ember",
                     "config": CONFIG, "model": model,
                     "text": res["text"], "finish": res["finish"],
                     "prompt_tokens": res["prompt_tokens"],
                     "completion_tokens": res["completion_tokens"],
                     "reasoning_tokens": res["reasoning_tokens"],
                     "computed_cost": cost,
                     "latency_s": res["latency_s"]})
        print(f"{key} in={res['prompt_tokens']} "
              f"out={res['completion_tokens']} "
              f"(reason={res['reasoning_tokens']}) "
              f"${cost:.5f} {res['latency_s']}s", flush=True)
    minutes = (time.time() - t0) / 60
    ok = [r for r in rows if "error" not in r]
    with open(RUN_DIR / "judgments_meta.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"config": CONFIG, "model": model, "calls": len(ok),
            "errors": len(rows) - len(ok),
            "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
            "completion_tokens": sum(r["completion_tokens"] for r in ok),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in ok),
            "computed_cost": sum(r["computed_cost"] for r in ok),
            "minutes": minutes}
    with open(RUN_DIR / "judge_spend_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


def _t5_wins(parse: dict[str, dict]) -> dict:
    wins = {"dpo": 0, "base": 0, "tie": 0, "inconsistent": 0}
    pairs: dict[str, dict[str, str]] = {}
    for key, verdict in parse.items():
        _, order, pid = key.split(":", 2)
        pairs.setdefault(pid, {})[order] = verdict.get("verdict", "")
    for pid, orders in pairs.items():
        o, s = orders.get("orig", ""), orders.get("swap", "")
        flip = {"A": "B", "B": "A", "C": "C"}
        if o not in flip or s != flip[o]:
            wins["inconsistent"] += 1
        elif o == "C":
            wins["tie"] += 1
        elif o == "B":
            wins["dpo"] += 1
        else:
            wins["base"] += 1
    n = len(pairs)
    return {"pairs": n, **wins,
            "dpo_win_rate": wins["dpo"] / n if n else None,
            "dpo_win_ci": wilson(wins["dpo"], n)}


def cmd_analyze() -> None:
    spend = json.load(open(RUN_DIR / "judge_spend.json"))
    judge_a, judge_b = spend["judges"]
    hf: dict[str, dict[str, dict]] = {}
    for line in open(RUN_DIR / "judgments.jsonl"):
        row = json.loads(line)
        hf.setdefault(row["key"], {})[row["judge"]] = row
    meta: dict[str, dict] = {}
    for line in open(RUN_DIR / "judgments_meta.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            meta[row["key"]] = row

    summary: dict = {"judges": [judge_a, judge_b, "meta-ember"],
                     "single": {}, "t5": {}}
    for kind, labels in TRACK_LABELS.items():
        keys = sorted({k for k in hf if k.startswith(kind + ":")
                       and k in meta})
        series = {}
        for name, get in (
                (judge_a, lambda k: hf[k][judge_a]["text"]),
                (judge_b, lambda k: hf[k][judge_b]["text"]),
                ("meta-ember", lambda k: meta[k]["text"])):
            series[name] = [parse_judgment(get(k), labels)["verdict"]
                            or "none" for k in keys]
        names = list(series)
        pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = series[names[i]], series[names[j]]
                agree = sum(1 for x, y in zip(a, b) if x == y)
                pairs[f"{names[i]}-vs-{names[j]}"] = {
                    "agree": agree, "n": len(keys),
                    "agree_ci": wilson(agree, len(keys)),
                    "kappa": cohen_kappa(a, b) if keys else None,
                }
        summary["single"][kind] = {
            "n": len(keys), "pairs": pairs,
            "bar_pass": all(p["agree"] / p["n"] >= 0.80
                            and (p["kappa"] or 0) >= 0.60
                            for p in pairs.values()) if keys else None,
        }

    for name, texts in (
            (judge_a, {k: parse_pairwise(hf[k][judge_a]["text"])
                       for k in hf if k.startswith("T5:")}),
            ("meta-ember", {k: parse_pairwise(meta[k]["text"])
                            for k in meta if k.startswith("T5:")})):
        summary["t5"][name] = _t5_wins(texts)
    with open(RUN_DIR / "agreement3.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.load(open(RUN_DIR / "agreement3.json"))
    kinds = [k for k in ("T2", "T3", "T4", "T1") if k in summary["single"]]
    pair_names = list(next(iter(summary["single"].values()))["pairs"])
    x = range(len(kinds))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for idx, pname in enumerate(pair_names):
        vals, lo, hi = [], [], []
        for kind in kinds:
            cell = summary["single"][kind]["pairs"][pname]
            v = cell["agree"] / cell["n"]
            ci_lo, ci_hi = cell["agree_ci"]
            vals.append(v)
            lo.append(v - ci_lo)
            hi.append(ci_hi - v)
        off = (idx - (len(pair_names) - 1) / 2) * width
        ax.bar([i + off for i in x], vals, width, yerr=[lo, hi],
               capsize=2, label=pname)
    ax.set_xticks(list(x), kinds)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.80, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("verdict agreement (Wilson 95%)")
    ax.set_title("Three-judge agreement per track (#74: HF pair + Meta)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "issue-74-agreement3.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {REPORT_DIR / 'issue-74-agreement3.png'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="issue-74 meta judge")
    parser.add_argument("command",
                        choices=["estimate", "run", "analyze", "plot"])
    args = parser.parse_args(argv)
    {"estimate": cmd_estimate, "run": cmd_run,
     "analyze": cmd_analyze, "plot": cmd_plot}[args.command]()


if __name__ == "__main__":
    main()
