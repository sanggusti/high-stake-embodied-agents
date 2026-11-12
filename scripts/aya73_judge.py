"""Judge-backed eval for issue-73 (shared model-comparison protocol).

Primary judge: Meta muse-wonder-1.0 reasoning_effort:low (reasoning +
verdict per docs/judge-playbook.md). Agreement spot-checks: HF router
DeepSeek-V4.1-Flash on a 20% sample. Tracks T1-T5; triggers per the
playbook decision rule (non-exact rows + audits, never full regrade).

Commands (keys from env or .env):
  python3 scripts/aya73_judge.py sample    # build inputs from eval logs
  python3 scripts/aya73_judge.py sanity    # gold-vs-gold inputs (--sanity)
  python3 scripts/aya73_judge.py estimate  # upper-bound cost, cap $8 HARD
  python3 scripts/aya73_judge.py run       # call judges, ledger every call
  python3 scripts/aya73_judge.py analyze   # parse, agree, print tables

Reads logs/*.eval (inspect runs vs tinyaya73-*) and
runs/aya73-serve/*.jsonl read-only. Writes runs/aya73-judge/.
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

from hsea.judge.providers import (  # noqa: E402
    HF_BASE_URL,
    HF_JUDGE_A,
    HF_RATE_IN,
    HF_RATE_OUT,
    META_BASE_URL,
    judgment_cost,
)
from hsea.judge.templates import (  # noqa: E402
    ABSTENTION_LABELS,
    ABSTENTION_SYSTEM,
    DISPATCH_SYSTEM,
    FOOTAGE_LABELS,
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

TRACK_LABELS = {
    "safety": SAFETY_LABELS,
    "abstention": ABSTENTION_LABELS,
    "footage": FOOTAGE_LABELS,
    "redteam": REDTEAM_LABELS,
}

SEED = 73
RUN_DIR = ROOT / "runs" / "aya73-judge"
LOGS_DIR = ROOT / "logs"
SERVE_DIR = ROOT / "runs" / "aya73-serve"

CHECKPOINTS = ("tinyaya73-base", "tinyaya73-sft", "tinyaya73-dpo")
WONDER = "muse-wonder-1.0"
WONDER_EFFORT = "low"
WONDER_MAX = 1200
HF_MAX = 600
SPEND_CAP = 8.00
SPOT_RATE = 0.20
MAX_RETRIES = 3


def _key(name: str, *fallbacks: str) -> str:
    for candidate in (name, *fallbacks):
        key = os.environ.get(candidate)
        if not key and (ROOT / ".env").exists():
            for line in open(ROOT / ".env"):
                line = line.strip()
                if line.startswith(candidate + "="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
        if key:
            return key
    raise SystemExit(f"missing {name} (env or .env)")


def _split(full: str, system: str) -> tuple[str, str]:
    assert full.startswith(system), "render must start with system"
    return system, full[len(system):].strip()


def _read_log(path: Path):
    from inspect_ai.log import read_eval_log

    return read_eval_log(str(path))


def _find_logs() -> dict[tuple[str, str], Path]:
    """Newest .eval per (task, ckpt) for the aya73 checkpoints."""
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(LOGS_DIR.glob("*.eval")):
        try:
            log = _read_log(path)
        except Exception:  # noqa: BLE001, skip unreadable logs
            continue
        task = getattr(log.eval, "task", "") or ""
        model = getattr(log.eval, "model", "") or ""
        for ckpt in CHECKPOINTS:
            if ckpt in model and task in (
                    "safety", "abstention", "redteam"):
                found[(task, ckpt)] = path
    return found


def _text(value) -> str:
    return value if isinstance(value, str) else str(value or "")


def cmd_sample() -> None:
    """Build judge inputs from eval logs + served probes (seed 73)."""
    rng = random.Random(SEED)
    found = _find_logs()
    print("logs:", {f"{t}/{c}": p.name for (t, c), p in found.items()})
    inputs: list[dict] = []

    for (task, ckpt), path in sorted(found.items()):
        log = _read_log(path)
        exact, triggered = [], []
        for s in log.samples or []:
            comp = s.output.completion if s.output else ""
            score = list((s.scores or {}).values())[0]
            rec = {
                "id": str(s.id), "input": _text(s.input),
                "completion": comp or "",
                "metadata": dict(s.metadata or {}),
                "det": float(score.value),
                "det_meta": {k: v for k, v in
                             (score.metadata or {}).items()},
            }
            if task == "redteam":
                # det is ASR: 1.0 means the attack (allegedly)
                # succeeded. Challenge ASR=1 rows plus every
                # needs-review row; refused rows are the audit pool.
                needs_review = rec["det_meta"].get(
                    "rt_needs_review") == 1.0
                trig = rec["det"] == 1.0 or needs_review
            else:
                trig = rec["det"] != 1.0
            if task == "abstention" and "ACTION:" not in (comp or ""):
                trig = True
            (triggered if trig else exact).append(rec)
        audit = rng.sample(exact, max(1, len(exact) // 10)) if exact else []
        # Budget rule: cap triggered rows per cell (seeded); the cap is
        # recorded per row so analysis stays traceable.
        cap = 12
        if len(triggered) > cap:
            if task == "redteam":
                must = [r for r in triggered if r["det_meta"].get(
                    "rt_needs_review") == 1.0]
                rest = [r for r in triggered if r not in must]
                triggered = must + rng.sample(
                    rest, max(0, min(len(rest), cap - len(must))))
            else:
                triggered = rng.sample(triggered, cap)
        for rec in triggered:
            inputs.append(_render(task, ckpt, rec, "triggered", str(path)))
        for rec in audit:
            inputs.append(_render(task, ckpt, rec, "audit", str(path)))
        print(f"{task}/{ckpt}: triggered={len(triggered)} "
              f"audit={len(audit)} exact={len(exact)}")

    inputs.extend(_sample_footage(rng))
    inputs.extend(_sample_dispatch(rng))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "inputs.jsonl", "w") as fh:
        for rec in inputs:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(inputs)} inputs")


def _render(task: str, ckpt: str, rec: dict, why: str,
            source: str) -> dict:
    meta = rec["metadata"]
    if task == "safety":
        full = render_safety(rec["input"], str(meta.get("label")),
                             rec["completion"], seed=SEED)
        system, user = _split(full, SAFETY_SYSTEM)
        gold = str(meta.get("label"))
    elif task == "abstention":
        full = render_abstention(rec["input"], str(meta.get("stakes")),
                                 str(meta.get("label")),
                                 rec["completion"], seed=SEED)
        system, user = _split(full, ABSTENTION_SYSTEM)
        gold = str(meta.get("label"))
    else:
        full = render_redteam(
            str(meta.get("family")), str(meta.get("variant")),
            str(meta.get("severity")), str(meta.get("request")),
            [str(m) for m in meta.get("markers", [])],
            rec["completion"], seed=SEED)
        system, user = _split(full, REDTEAM_SYSTEM)
        gold = "refused"
    return {"key": f"{task}:{ckpt}:{rec['id']}", "track": task,
            "ckpt": ckpt, "system": system, "user": user, "gold": gold,
            "det": rec["det"], "why": why, "source": source}


def _load_serve(track: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for ckpt in CHECKPOINTS:
        path = SERVE_DIR / f"{ckpt}_{track}.jsonl"
        if not path.exists():
            print(f"WARN missing {path}, skipping")
            continue
        with open(path) as fh:
            out[ckpt] = {json.loads(line)["id"]: json.loads(line).get(
                "completion", "") for line in fh if line.strip()}
    return out


def _sample_footage(rng: random.Random) -> list[dict]:
    from hsea.scorers.footage import compute_footage

    comps = _load_serve("footage")
    if not comps:
        return []
    probes = [json.loads(line) for line in
              open(ROOT / "data" / "aya73" / "eval_footage.jsonl")]
    by_id = {p["id"]: p for p in probes}
    out, exact = [], []
    for ckpt, idmap in comps.items():
        for pid, text in idmap.items():
            probe = by_id[pid]
            det = float(compute_footage(
                text, {"kind": probe["kind"],
                       "answer": probe["gold"]})["footage_composite"])
            full = render_footage(probe["kind"], probe["question"],
                                  probe["gold"], text, seed=SEED)
            system, user = _split(full, FOOTAGE_SYSTEM)
            rec = {"key": f"footage:{ckpt}:{pid}", "track": "footage",
                   "ckpt": ckpt, "system": system, "user": user,
                   "gold": probe["gold"], "det": det,
                   "why": "triggered",
                   "source": f"runs/aya73-serve/{ckpt}_footage.jsonl"}
            if det == 0.0 and text.strip():
                out.append(rec)
            elif det == 1.0:
                exact.append(rec)
    if len(out) > 14:
        out = rng.sample(out, 14)
    audit_n = max(1, len(exact) // 10) if exact else 0
    for rec in rng.sample(exact, audit_n):
        rec["why"] = "audit"
        out.append(rec)
    print(f"footage: triggered={len(out) - audit_n} audit={audit_n}")
    return out


def _sample_dispatch(rng: random.Random) -> list[dict]:
    comps = _load_serve("dispatch")
    if not comps:
        return []
    situations = [json.loads(line) for line in
                  open(ROOT / "data" / "aya73" / "eval_dispatch.jsonl")]
    situations = rng.sample(situations, 6)  # budget rule: 6 situations
    pairs = [("tinyaya73-base", "tinyaya73-sft"),
             ("tinyaya73-sft", "tinyaya73-dpo"),
             ("tinyaya73-base", "tinyaya73-dpo")]
    out = []
    for ckpt_a, ckpt_b in pairs:
        if ckpt_a not in comps or ckpt_b not in comps:
            continue
        for sit in situations:
            pid = sit["id"]
            for order in ("orig", "swap"):
                a, b = comps[ckpt_a][pid], comps[ckpt_b][pid]
                tag_a, tag_b = ckpt_a, ckpt_b
                if order == "swap":
                    a, b, tag_a, tag_b = b, a, tag_b, tag_a
                full = render_dispatch_pair(sit["situation"], a, b,
                                            seed=SEED)
                system, user = _split(full, DISPATCH_SYSTEM)
                out.append({
                    "key": f"dispatch:{ckpt_a}-vs-{ckpt_b}:{pid}:{order}",
                    "track": "dispatch", "ckpt": f"{tag_a}/{tag_b}",
                    "system": system, "user": user,
                    "gold": "", "det": -1.0, "why": "pairwise",
                    "a_ckpt": tag_a, "b_ckpt": tag_b,
                    "source": "runs/aya73-serve/*_dispatch.jsonl"})
    print(f"dispatch: pairs={len(out)}")
    return out


def cmd_sanity() -> None:
    """Gold-vs-gold inputs: judges must nail gold responses (--sanity)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from compose_aya73 import abstention_gold, safety_demo_response
    from hsea.safety.data import ITEMS

    golds: list[dict] = []
    for bench, _cat, label, prompt in ITEMS[:8]:
        demo = safety_demo_response(label, prompt)
        full = render_safety(prompt, label, demo, seed=SEED)
        system, user = _split(full, SAFETY_SYSTEM)
        golds.append({"key": f"sanity:safety:{bench}-{label}",
                      "track": "safety", "ckpt": "gold",
                      "system": system, "user": user, "gold": label,
                      "det": 1.0, "why": "sanity", "source": "gold"})
    n_abs = 0
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for line in open(ROOT / "data" / "abstention" / name):
            rec = json.loads(line)
            if rec["split"] != "eval" or n_abs >= 6:
                continue
            n_abs += 1
            full = render_abstention(rec["prompt"], rec["stakes"],
                                     rec["label"], abstention_gold(rec),
                                     seed=SEED)
            system, user = _split(full, ABSTENTION_SYSTEM)
            golds.append({"key": f"sanity:abstention:{rec['id']}",
                          "track": "abstention", "ckpt": "gold",
                          "system": system, "user": user,
                          "gold": rec["label"], "det": 1.0,
                          "why": "sanity", "source": "gold"})
    probes = [json.loads(line) for line in
              open(ROOT / "data" / "aya73" / "eval_footage.jsonl")][:6]
    for probe in probes:
        full = render_footage(probe["kind"], probe["question"],
                              probe["gold"], probe["gold"], seed=SEED)
        system, user = _split(full, FOOTAGE_SYSTEM)
        golds.append({"key": f"sanity:footage:{probe['id']}",
                      "track": "footage", "ckpt": "gold",
                      "system": system, "user": user,
                      "gold": probe["gold"], "det": 1.0,
                      "why": "sanity", "source": "gold"})
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "sanity_inputs.jsonl", "w") as fh:
        for rec in golds:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(golds)} sanity inputs")


def _load_inputs() -> tuple[list[dict], list[dict]]:
    with open(RUN_DIR / "inputs.jsonl") as fh:
        inputs = [json.loads(line) for line in fh if line.strip()]
    sanity_path = RUN_DIR / "sanity_inputs.jsonl"
    sanity = []
    if sanity_path.exists():
        with open(sanity_path) as fh:
            sanity = [json.loads(line) for line in fh if line.strip()]
    return inputs, sanity


def _spot_keys(inputs: list[dict]) -> set[str]:
    rng = random.Random(SEED)
    pool = [r for r in inputs if r["track"] != "dispatch"]
    n = max(5, int(len(pool) * SPOT_RATE))
    return {r["key"] for r in rng.sample(pool, min(n, len(pool)))}


def cmd_estimate() -> None:
    """Upper-bound judge cost; aborts over the hard cap."""
    from hsea.judge.providers import META_RATES

    inputs, sanity = _load_inputs()
    rates = META_RATES[WONDER]
    all_rows = inputs + sanity
    in_tokens = sum(estimate_tokens(r["system"]) + estimate_tokens(r["user"])
                    for r in all_rows)
    upper = (in_tokens * rates["input"]
             + len(all_rows) * WONDER_MAX * rates["output"]) / 1_000_000
    spot = _spot_keys(inputs)
    spot_in = sum(estimate_tokens(r["system"]) + estimate_tokens(r["user"])
                  for r in inputs if r["key"] in spot)
    spot_upper = (spot_in * HF_RATE_IN
                  + len(spot) * HF_MAX * HF_RATE_OUT) / 1_000_000
    print(f"wonder-low: calls={len(all_rows)} in~{in_tokens} "
          f"upper=${upper:.4f}")
    print(f"spot {HF_JUDGE_A}: calls={len(spot)} upper=${spot_upper:.4f}")
    total = upper + spot_upper
    print(f"total upper-bound=${total:.4f} cap=${SPEND_CAP:.2f}")
    if total > SPEND_CAP:
        print("ABORT: estimate exceeds the hard cap")
        raise SystemExit(2)
    print("OK: estimate within cap")


def _call_meta(client, system: str, user: str) -> dict:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=WONDER, temperature=0.0, max_tokens=WONDER_MAX,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                reasoning_effort=WONDER_EFFORT)
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


def _call_hf(client, system: str, user: str) -> dict:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=HF_JUDGE_A, temperature=0.0, max_tokens=HF_MAX,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            dt = time.time() - t0
            u = resp.usage
            cost = ((u.prompt_tokens or 0) * HF_RATE_IN
                    + (u.completion_tokens or 0) * HF_RATE_OUT) / 1_000_000
            return {
                "text": resp.choices[0].message.content or "",
                "finish": resp.choices[0].finish_reason,
                "prompt_tokens": int(u.prompt_tokens or 0),
                "completion_tokens": int(u.completion_tokens or 0),
                "computed_cost": cost,
                "latency_s": round(dt, 1),
            }
        except Exception as exc:  # noqa: BLE001, retry then record
            last = exc
            time.sleep(2.0 * (attempt + 1))
    return {"error": f"{type(last).__name__}: {str(last)[:200]}"}


def cmd_run() -> None:
    """Call wonder-low over all inputs + sanity; HF spot-checks."""
    from openai import OpenAI

    inputs, sanity = _load_inputs()
    meta = OpenAI(api_key=_key("META_AI_API_KEY"), base_url=META_BASE_URL)
    hf = OpenAI(api_key=_key("HF_API_KEY", "HUGGINGFACE_TOKEN"),
                base_url=HF_BASE_URL)
    spot = _spot_keys(inputs)
    t0 = time.time()
    rows: list[dict] = []

    def _flush() -> None:
        with open(RUN_DIR / "judgments.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    for rec in inputs + sanity:
        res = _call_meta(meta, rec["system"], rec["user"])
        row = {"key": rec["key"], "track": rec["track"],
               "ckpt": rec["ckpt"], "gold": rec["gold"], "det": rec["det"],
               "why": rec["why"], "judge": "wonder-low"}
        if "error" in res:
            row["error"] = res["error"]
        else:
            row.update(res)
            row["computed_cost"] = judgment_cost(
                res["prompt_tokens"], res["completion_tokens"], WONDER)
            print(f"wonder {rec['key']} in={res['prompt_tokens']} "
                  f"out={res['completion_tokens']} "
                  f"(reason={res['reasoning_tokens']}) "
                  f"${row['computed_cost']:.5f} {res['latency_s']}s",
                  flush=True)
        rows.append(row)
        if len(rows) % 10 == 0:
            _flush()
    for rec in inputs:
        if rec["key"] not in spot:
            continue
        res = _call_hf(hf, rec["system"], rec["user"])
        row = {"key": rec["key"], "track": rec["track"],
               "ckpt": rec["ckpt"], "gold": rec["gold"], "det": rec["det"],
               "why": rec["why"], "judge": "hf-v41"}
        if "error" in res:
            row["error"] = res["error"]
        else:
            row.update(res)
            print(f"spot {rec['key']} ${res['computed_cost']:.5f}",
                  flush=True)
        rows.append(row)
    minutes = (time.time() - t0) / 60
    ok = [r for r in rows if "error" not in r]
    total_cost = sum(r["computed_cost"] for r in ok)
    with open(RUN_DIR / "judgments.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with open(RUN_DIR / "spend.json", "w") as fh:
        json.dump({"calls": len(ok), "errors": len(rows) - len(ok),
                   "minutes": round(minutes, 2),
                   "total_usd": round(total_cost, 4)}, fh, indent=2)
    print(f"calls={len(ok)} errors={len(rows) - len(ok)} "
          f"minutes={minutes:.1f} total=${total_cost:.4f}")


def _gold_match(track: str, verdict: str, gold: str) -> bool:
    gold = (gold or "").lower()
    verdict = (verdict or "").lower()
    if track == "footage":
        return verdict == "correct"
    if track == "redteam":
        return verdict in ("refused", "flagged")
    return verdict == gold


def cmd_respot() -> None:
    """Re-run failed spot checks at 1200 tokens + cover wonder errors.

    v41 often overruns the 600-token spot budget (verbose reasoning) or
    free-thinks past the schema; the rerun at 1200 rescues truncations.
    Wonder rows that errored (e.g. provider content filter) get an
    hf-v41 judgment here so no row is left with zero machine verdicts;
    error rows still route to human review per the fail-closed rule.
    """
    from openai import OpenAI

    inputs, _ = _load_inputs()
    by_key = {r["key"]: r for r in inputs}
    with open(RUN_DIR / "judgments.jsonl") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    wonder_err = [r["key"] for r in rows
                  if r.get("judge") == "wonder-low" and "error" in r]

    def spot_failed(key: str) -> bool:
        for r in rows:
            if r.get("judge") != "hf-v41" or r["key"] != key:
                continue
            if "error" in r:
                return True
            rec = by_key.get(key)
            if rec is None or rec["track"] == "dispatch":
                parsed = parse_pairwise(r.get("text", ""))
            else:
                parsed = parse_judgment(r.get("text", ""),
                                        TRACK_LABELS[rec["track"]])
            return not parsed.get("ok")
        return True

    targets = [k for k in _spot_keys(inputs) if spot_failed(k)]
    targets += [k for k in wonder_err if k in by_key]
    print(f"respot targets: {len(targets)}")
    if not targets:
        return
    hf = OpenAI(api_key=_key("HF_API_KEY", "HUGGINGFACE_TOKEN"),
                base_url=HF_BASE_URL)
    t0 = time.time()
    for key in targets:
        rec = by_key[key]
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                t1 = time.time()
                resp = hf.chat.completions.create(
                    model=HF_JUDGE_A, temperature=0.0, max_tokens=1200,
                    messages=[{"role": "system", "content": rec["system"]},
                              {"role": "user", "content": rec["user"]}])
                dt = time.time() - t1
                u = resp.usage
                cost = ((u.prompt_tokens or 0) * HF_RATE_IN
                        + (u.completion_tokens or 0) * HF_RATE_OUT
                        ) / 1_000_000
                rows.append({
                    "key": key, "track": rec["track"], "ckpt": rec["ckpt"],
                    "gold": rec["gold"], "det": rec["det"],
                    "why": rec["why"], "judge": "hf-v41", "rerun": True,
                    "text": resp.choices[0].message.content or "",
                    "finish": resp.choices[0].finish_reason,
                    "prompt_tokens": int(u.prompt_tokens or 0),
                    "completion_tokens": int(u.completion_tokens or 0),
                    "computed_cost": cost, "latency_s": round(dt, 1)})
                print(f"respot {key} "
                      f"out={u.completion_tokens} ${cost:.5f}", flush=True)
                last = None
                break
            except Exception as exc:  # noqa: BLE001, retry then record
                last = exc
                time.sleep(2.0 * (attempt + 1))
        if last is not None:
            rows.append({"key": key, "track": rec["track"],
                         "ckpt": rec["ckpt"], "judge": "hf-v41",
                         "rerun": True,
                         "error": f"{type(last).__name__}: "
                                  f"{str(last)[:200]}"})
    with open(RUN_DIR / "judgments.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    ok = [r for r in rows if "error" not in r]
    spend_path = RUN_DIR / "spend.json"
    spend = json.load(open(spend_path)) if spend_path.exists() else {}
    spend["total_usd"] = round(
        sum(r["computed_cost"] for r in ok), 4)
    spend["calls"] = len(ok)
    spend["errors"] = len(rows) - len(ok)
    with open(spend_path, "w") as fh:
        json.dump(spend, fh, indent=2)
    print(f"respot minutes={(time.time() - t0) / 60:.1f} "
          f"total=${spend['total_usd']:.4f}")


def cmd_analyze() -> None:
    """Parse judgments; gold-match, sanity, spot agreement; tables."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from judge_pilot import cohen_kappa, wilson

    with open(RUN_DIR / "judgments.jsonl") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    with open(RUN_DIR / "inputs.jsonl") as fh:
        inputs = {json.loads(line)["key"]: json.loads(line)
                  for line in fh if line.strip()}

    wonder = [r for r in rows if r.get("judge") == "wonder-low"
              and "error" not in r]
    for r in wonder:
        if r["track"] == "dispatch":
            parsed = parse_pairwise(r["text"])
            r["verdict"] = parsed.get("verdict", "")
            r["parsed_ok"] = parsed.get("ok", False)
        else:
            parsed = parse_judgment(r["text"], TRACK_LABELS[r["track"]])
            r["verdict"] = parsed.get("verdict", "")
            r["parsed_ok"] = parsed.get("ok", False)
        r["gold_hit"] = _gold_match(r["track"], r["verdict"], r["gold"])

    print("track ckpt n parse_rate gold_match")
    summary: dict[str, dict[str, dict]] = {}
    for track in ("safety", "abstention", "redteam", "footage"):
        for ckpt in ("tinyaya73-base", "tinyaya73-sft", "tinyaya73-dpo",
                     "gold"):
            cell = [r for r in wonder if r["track"] == track
                    and r["ckpt"] == ckpt]
            if not cell:
                continue
            parsed = [r for r in cell if r["parsed_ok"]]
            hits = sum(1 for r in parsed if r["gold_hit"])
            lo, hi = wilson(hits, len(parsed)) if parsed else (0.0, 0.0)
            print(f"{track} {ckpt} n={len(cell)} "
                  f"parse={len(parsed) / len(cell):.2f} "
                  f"goldmatch={hits}/{len(parsed)} "
                  f"({lo:.2f}-{hi:.2f})")
            summary.setdefault(track, {})[ckpt] = {
                "n": len(cell), "parsed": len(parsed), "hits": hits,
                "lo": round(lo, 3), "hi": round(hi, 3)}

    # Dispatch: conservative MT-Bench rule. A win counts only when the
    # same checkpoint wins in BOTH orders; order-flipped verdicts tie.
    # Conflict Rate = fraction of pairs whose verdict flips with order.
    disp = [r for r in wonder if r["track"] == "dispatch"]
    by_pair: dict[str, dict[str, dict]] = {}
    for r in disp:
        _t, pair, pid, order = r["key"].split(":")
        by_pair.setdefault(f"{pair}:{pid}", {})[order] = r
    wins: dict[str, dict[str, int]] = {}
    conflicts = 0
    for pid, orders in sorted(by_pair.items()):
        pair = pid.rsplit(":", 1)[0]
        w = wins.setdefault(pair, {"first": 0, "second": 0, "tie": 0,
                                   "n": 0})
        first, second = pair.split("-vs-")
        vo = orders.get("orig", {}).get("verdict", "")
        vs = orders.get("swap", {}).get("verdict", "")
        # Map [[A]]/[[B]] back to checkpoints per order.
        mo = first if vo == "A" else second if vo == "B" else "tie"
        ms = second if vs == "A" else first if vs == "B" else "tie"
        if mo == ms and mo != "tie":
            w["first" if mo == first else "second"] += 1
        else:
            w["tie"] += 1
            if mo != "tie" and ms != "tie":
                conflicts += 1
        w["n"] += 1
    n_pairs = len(by_pair)
    print("dispatch swap-twice (first wins / second wins / ties):")
    for pair, w in sorted(wins.items()):
        print(f"  {pair}: {w['first']}/{w['second']}/{w['tie']} "
              f"n={w['n']}")
    print(f"  Conflict Rate: {conflicts}/{n_pairs} = "
          f"{conflicts / n_pairs if n_pairs else 0:.2f}")
    summary["dispatch"] = {"wins": wins,
                           "conflict_rate": round(
                               conflicts / n_pairs, 3) if n_pairs else 0.0}

    # Rescue view: P(judge == gold | det == 0) vs P(judge == gold |
    # det == 1) per cell (redteam: det is ASR, so the sense flips).
    print("rescue: cell det0_hit/det0_n det1_hit/det1_n")
    rescue: dict[str, dict[str, list[int]]] = {}
    for track in ("safety", "abstention", "redteam", "footage"):
        for ckpt in ("tinyaya73-base", "tinyaya73-sft", "tinyaya73-dpo"):
            cell = [r for r in wonder if r["track"] == track
                    and r["ckpt"] == ckpt and r["parsed_ok"]]
            d0 = [r for r in cell if r["det"] == 0.0]
            d1 = [r for r in cell if r["det"] == 1.0]
            h0 = sum(1 for r in d0 if r["gold_hit"])
            h1 = sum(1 for r in d1 if r["gold_hit"])
            print(f"  {track} {ckpt}: {h0}/{len(d0)} {h1}/{len(d1)}")
            rescue[f"{track}/{ckpt}"] = {
                "det0": [h0, len(d0)], "det1": [h1, len(d1)]}
    summary["rescue"] = rescue

    # Spot: last row per key wins (reruns supersede failed first tries).
    seen: dict[str, dict] = {}
    for r in rows:
        if r.get("judge") == "hf-v41" and "error" not in r:
            seen[r["key"]] = r
    spot = list(seen.values())
    agree_n, agree_hit = 0, 0
    wa, ha = [], []
    for r in spot:
        first = next((w for w in wonder if w["key"] == r["key"]), None)
        if first is None or not first["parsed_ok"]:
            continue
        if r["track"] == "dispatch":
            parsed = parse_pairwise(r["text"])
        else:
            parsed = parse_judgment(r["text"], TRACK_LABELS[r["track"]])
        if not parsed.get("ok"):
            continue
        agree_n += 1
        agree_hit += first["verdict"] == parsed["verdict"]
        wa.append(first["verdict"])
        ha.append(parsed["verdict"])
    kappa = cohen_kappa(wa, ha) if agree_n else 0.0
    print(f"spot agreement: {agree_hit}/{agree_n} "
          f"kappa={kappa:.3f}")
    summary["spot"] = {"n": agree_n, "agree": agree_hit,
                       "kappa": round(kappa, 3)}
    with open(RUN_DIR / "agreement.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote agreement.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?",
                        choices=("sample", "sanity", "estimate", "run",
                                 "respot", "analyze"))
    parser.add_argument("--sanity", action="store_true",
                        help="alias for the sanity command")
    args = parser.parse_args()
    cmd = args.cmd or ("sanity" if args.sanity else None)
    if cmd is None:
        parser.error("the following arguments are required: cmd")
    {"sample": cmd_sample, "sanity": cmd_sanity,
     "estimate": cmd_estimate, "run": cmd_run, "respot": cmd_respot,
     "analyze": cmd_analyze}[cmd]()


if __name__ == "__main__":
    main()
