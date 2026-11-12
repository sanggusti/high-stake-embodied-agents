"""Issue #8 judge grading: T1/T2/T4 over base-vs-ft completions.

Follows the judge playbook (#62): deterministic rates stay primary,
judges open only the flattened rows. Dual judges
(DeepSeek-V4.1-Flash + gemma-3-27b-it via the HF router) with a HARD
$2.00 spend cap: ``estimate`` aborts ``run`` when over budget.

Commands (run from the repo root, after issue8_evals.py)::

    .venv/bin/python scripts/issue8_judge.py sample
    .venv/bin/python scripts/issue8_judge.py estimate
    .venv/bin/python scripts/issue8_judge.py run
    .venv/bin/python scripts/issue8_judge.py analyze

Reads ``runs/issue-8/eval_manifest.json`` logs read-only; writes
``runs/issue-8/`` (inputs, judgments, agreement, spend).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hsea.judge.templates import (  # noqa: E402
    FOOTAGE_SYSTEM,
    REDTEAM_LABELS,
    REDTEAM_SYSTEM,
    SAFETY_LABELS,
    SAFETY_SYSTEM,
    estimate_tokens,
    parse_judgment,
    render_footage,
    render_redteam,
    render_safety,
)
from judge_pilot import JUDGE_A, JUDGE_B, cohen_kappa  # noqa: E402
from hsea.judge.providers import (  # noqa: E402
    META_BASE_URL,
    META_JUDGE_CONFIGS,
    META_RATES,
)

SEED = 8
RUN_DIR = ROOT / "runs" / "issue-8"
INPUTS = RUN_DIR / "judge_inputs.jsonl"
JUDGMENTS = RUN_DIR / "judgments.jsonl"
SUMMARY = RUN_DIR / "judge_summary.json"
SPEND = RUN_DIR / "judge_spend.json"
META_JUDGMENTS = RUN_DIR / "judgments_meta.jsonl"
META_SPEND = RUN_DIR / "judge_spend_meta.json"
META_CONFIG = "ember-minimal"
META_SPEND_CAP = 2.00
META_MAX_RETRIES = 3

HF_BASE_URL = "https://router.huggingface.co/v1"
RATE_IN = 0.30
RATE_OUT = 1.20
SPEND_CAP = 2.00
OUT_TOKENS_BOUND = 600
CHALLENGE_ASR1 = 5  # ASR=1 rows per model challenged by T4


def _flatten(log_path: str) -> list[dict]:
    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    log = read_eval_log(log_path)
    rows = []
    for s in log.samples or []:
        score = list((s.scores or {}).values())[0] if s.scores else None
        rows.append({
            "id": str(s.id),
            "input": s.input if isinstance(s.input, str) else str(s.input),
            "completion": (s.output.completion or "")
            if s.output is not None else "",
            "target": str(getattr(s.target, "text", s.target) or ""),
            "metadata": dict(s.metadata or {}),
            "value": float(score.value) if score is not None else 0.0,
            "score_metadata": dict(getattr(score, "metadata", None) or {})
            if score else {},
        })
    return rows


def cmd_sample() -> None:
    """Seed-8 judge inputs: flattened rows + audits + challenge sample."""
    rng = random.Random(SEED)
    manifest = json.loads((RUN_DIR / "eval_manifest.json").read_text())
    inputs: list[dict] = []
    for key, run in sorted(manifest["runs"].items()):
        model, alias = key.split("/", 1)
        rows = _flatten(run["log"])
        if alias in ("floodnet_vqa", "ladi_v2"):
            bad = [r for r in rows if r["value"] < 1.0 and r["completion"]]
            good = [r for r in rows if r["value"] == 1.0]
            rng.shuffle(good)
            audit = good[: max(1, len(good) // 10)] if good else []
            for r in bad + audit:
                inputs.append({
                    "template": "T1", "model": model, "task": alias,
                    "id": r["id"], "kind": str(r["metadata"].get("kind", "vqa")),
                    "question": r["input"], "gold": r["target"],
                    "response": r["completion"], "det": r["value"],
                })
        elif alias == "safety":
            for r in rows:
                if r["value"] < 1.0:
                    inputs.append({
                        "template": "T2", "model": model, "task": alias,
                        "id": r["id"],
                        "request": r["input"],
                        "gold": str(r["metadata"].get("label", r["target"])),
                        "response": r["completion"], "det": r["value"],
                    })
        elif alias == "redteam":
            review = [r for r in rows
                      if float(r["score_metadata"].get("rt_needs_review", 0.0))]
            asr1 = [r for r in rows if r["value"] == 1.0]
            rng.shuffle(asr1)
            for r in review + asr1[:CHALLENGE_ASR1]:
                inputs.append({
                    "template": "T4", "model": model, "task": alias,
                    "id": r["id"],
                    "family": str(r["metadata"].get("family", "?")),
                    "variant": str(r["metadata"].get("variant", "?")),
                    "severity": str(r["metadata"].get("severity", "?")),
                    "request": str(r["metadata"].get("request", "")),
                    "markers": list(r["metadata"].get("markers", [])),
                    "response": r["completion"], "det": r["value"],
                })
    with open(INPUTS, "w") as fh:
        for rec in inputs:
            fh.write(json.dumps(rec) + "\n")
    by_template: dict[str, int] = {}
    for rec in inputs:
        by_template[rec["template"]] = by_template.get(rec["template"], 0) + 1
    print(f"inputs={len(inputs)} {by_template} -> {INPUTS}")


def _prompts() -> list[tuple[str, str, str, dict]]:
    """(key, system, user, rec) for every planned judgment (no API)."""

    def split(full: str, system: str) -> tuple[str, str]:
        assert full.startswith(system), "render must start with system"
        return system, full[len(system):].strip()

    out = []
    for line in open(INPUTS):
        rec = json.loads(line)
        if rec["template"] == "T1":
            full = render_footage(rec["kind"], rec["question"], rec["gold"],
                                  rec["response"], seed=SEED)
            system, user = split(full, FOOTAGE_SYSTEM)
        elif rec["template"] == "T2":
            full = render_safety(rec["request"], rec["gold"],
                                 rec["response"], seed=SEED)
            system, user = split(full, SAFETY_SYSTEM)
        else:
            full = render_redteam(rec["family"], rec["variant"],
                                  rec["severity"], rec["request"],
                                  rec["markers"], rec["response"], seed=SEED)
            system, user = split(full, REDTEAM_SYSTEM)
        key = f"{rec['template']}:{rec['model']}:{rec['task']}:{rec['id']}"
        out.append((key, system, user, rec))
    return out


def cmd_estimate() -> None:
    """Upper-bound cost; aborts when over the hard cap."""
    prompts = _prompts()
    calls = len(prompts) * 2  # dual judges
    in_tokens = sum(estimate_tokens(s) + estimate_tokens(u)
                    for _, s, u, _ in prompts) * 2
    out_tokens = calls * OUT_TOKENS_BOUND
    upper = (in_tokens * RATE_IN + out_tokens * RATE_OUT) / 1_000_000
    print(f"prompts={len(prompts)} calls={calls} "
          f"in~{in_tokens} out<={out_tokens}")
    print(f"upper-bound=${upper:.4f} cap=${SPEND_CAP:.2f}")
    if upper > SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _client():
    from openai import OpenAI

    key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_TOKEN")
    if not key:
        raise SystemExit("missing HF_API_KEY (or HUGGINGFACE_TOKEN)")
    return OpenAI(api_key=key, base_url=HF_BASE_URL)


def cmd_run() -> None:
    """Call both judges over all prompts; ledger every call."""
    prompts = _prompts()
    client = _client()
    t0 = time.time()
    rows: list[dict] = []
    for key, system, user, _rec in prompts:
        for judge in (JUDGE_A, JUDGE_B):
            resp = client.chat.completions.create(
                model=judge, temperature=0.0, max_tokens=OUT_TOKENS_BOUND,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            msg = resp.choices[0].message.content or ""
            u = resp.usage
            cost = None
            try:
                cost = float(getattr(u, "estimated_cost", None) or 0.0)
            except (TypeError, ValueError):
                cost = None
            rows.append({
                "key": key, "judge": judge, "text": msg,
                "prompt_tokens": int(u.prompt_tokens or 0),
                "completion_tokens": int(u.completion_tokens or 0),
                "estimated_cost": cost,
            })
            print(f"{key} {judge.split('/')[-1]} "
                  f"in={u.prompt_tokens} out={u.completion_tokens} "
                  f"${(cost or 0.0):.6f}", flush=True)
    minutes = (time.time() - t0) / 60
    total_cost = sum(r["estimated_cost"] or 0.0 for r in rows)
    with open(JUDGMENTS, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"calls": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "estimated_cost": total_cost, "minutes": minutes}
    SPEND.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


_LABELS = {"T1": ("correct", "close", "wrong"), "T2": SAFETY_LABELS,
           "T4": REDTEAM_LABELS}


def analyze_inputs(inputs: list[dict],
                   judgments: dict[str, dict[str, str]]) -> dict:
    """Pure agreement + outcome roll-up (tested, no I/O)."""
    by_template: dict[str, dict] = {}
    for rec in inputs:
        key = f"{rec['template']}:{rec['model']}:{rec['task']}:{rec['id']}"
        cell = by_template.setdefault(rec["template"], {
            "n": 0, "agree": 0, "pairs_a": [], "pairs_b": [],
            "parse_fail": 0, "by_model": {}})
        parsed = {}
        for judge in (JUDGE_A, JUDGE_B):
            parsed[judge] = parse_judgment(
                judgments.get(key, {}).get(judge, ""), _LABELS[rec["template"]])
        cell["n"] += 1
        if not parsed[JUDGE_A]["ok"] or not parsed[JUDGE_B]["ok"]:
            cell["parse_fail"] += 1
            continue
        va, vb = parsed[JUDGE_A]["verdict"], parsed[JUDGE_B]["verdict"]
        cell["pairs_a"].append(va)
        cell["pairs_b"].append(vb)
        if va == vb:
            cell["agree"] += 1
        slot = cell["by_model"].setdefault(rec["model"], {
            "n": 0, "verdicts": [], "scores": []})
        slot["n"] += 1
        slot["verdicts"].append(va)
        slot["scores"].append(parsed[JUDGE_A]["score"])
    summary = {}
    for template, cell in by_template.items():
        summary[template] = {
            "n": cell["n"],
            "agreement": (cell["agree"] / cell["n"]) if cell["n"] else 0.0,
            "kappa": cohen_kappa(cell["pairs_a"], cell["pairs_b"]),
            "parse_fail": cell["parse_fail"],
            "by_model": {
                m: {"n": s["n"],
                    "verdict_counts": {v: s["verdicts"].count(v)
                                       for v in sorted(set(s["verdicts"]))},
                    "mean_score": (sum(s["scores"]) / len(s["scores"])
                                   if s["scores"] else 0.0)}
                for m, s in sorted(cell["by_model"].items())
            },
        }
    return summary


def cmd_analyze() -> None:
    """Agreement + outcome roll-up into judge_summary.json."""
    inputs = [json.loads(line) for line in open(INPUTS)]
    judgments: dict[str, dict[str, str]] = {}
    for line in open(JUDGMENTS):
        row = json.loads(line)
        judgments.setdefault(row["key"], {})[row["judge"]] = row["text"]
    summary = analyze_inputs(inputs, judgments)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:4000])


def _meta_key() -> str:
    key = os.environ.get("META_AI_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in open(ROOT / ".env"):
            if line.startswith("META_AI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("missing META_AI_API_KEY (env or .env)")
    return key


def cmd_meta_estimate() -> None:
    """Upper-bound Meta cost over the same prompts; aborts over cap."""
    prompts = _prompts()
    cfg = META_JUDGE_CONFIGS[META_CONFIG]
    rates = META_RATES[str(cfg["model"])]
    in_tokens = sum(estimate_tokens(s) + estimate_tokens(u)
                    for _, s, u, _ in prompts)
    out_tokens = len(prompts) * int(cfg["max_tokens"])
    upper = (in_tokens * rates["input"]
             + out_tokens * rates["output"]) / 1_000_000
    print(f"config={META_CONFIG} prompts={len(prompts)} "
          f"in~{in_tokens} out<={out_tokens}")
    print(f"upper-bound=${upper:.4f} cap=${META_SPEND_CAP:.2f}")
    if upper > META_SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _meta_call(client, model: str, effort: str, max_out: int,
               system: str, user: str) -> dict:
    last: Exception | None = None
    for attempt in range(META_MAX_RETRIES):
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


def cmd_meta_run() -> None:
    """Call ember-minimal over all prompts; ledger every call."""
    from openai import OpenAI

    prompts = _prompts()
    cfg = META_JUDGE_CONFIGS[META_CONFIG]
    model = str(cfg["model"])
    effort = str(cfg["reasoning_effort"])
    max_out = int(cfg["max_tokens"])
    rates = META_RATES[model]
    client = OpenAI(api_key=_meta_key(), base_url=META_BASE_URL)
    t0 = time.time()
    rows: list[dict] = []
    for key, system, user, _rec in prompts:
        out = _meta_call(client, model, effort, max_out, system, user)
        if "error" in out:
            print(f"{key} ERROR {out['error']}", flush=True)
            rows.append({"key": key, "judge": META_CONFIG, "text": "",
                         "error": out["error"], "prompt_tokens": 0,
                         "completion_tokens": 0, "estimated_cost": 0.0})
            continue
        cost = (out["prompt_tokens"] * rates["input"]
                + out["completion_tokens"] * rates["output"]) / 1e6
        rows.append({"key": key, "judge": META_CONFIG, "text": out["text"],
                     "prompt_tokens": out["prompt_tokens"],
                     "completion_tokens": out["completion_tokens"],
                     "reasoning_tokens": out["reasoning_tokens"],
                     "estimated_cost": cost})
        print(f"{key} {META_CONFIG} in={out['prompt_tokens']} "
              f"out={out['completion_tokens']} "
              f"reason={out['reasoning_tokens']} ${cost:.6f}", flush=True)
    minutes = (time.time() - t0) / 60
    total_cost = sum(r["estimated_cost"] or 0.0 for r in rows)
    with open(META_JUDGMENTS, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"config": META_CONFIG, "calls": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "estimated_cost": total_cost, "minutes": minutes}
    META_SPEND.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["sample", "estimate", "run", "analyze",
                                 "meta-estimate", "meta-run"])
    args = parser.parse_args(argv)
    {"sample": cmd_sample, "estimate": cmd_estimate, "run": cmd_run,
     "analyze": cmd_analyze, "meta-estimate": cmd_meta_estimate,
     "meta-run": cmd_meta_run}[args.command]()


if __name__ == "__main__":
    main()
