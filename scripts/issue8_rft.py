"""Issue #8 E-RFT baseline: verifier-filtered SFT (stage 5, CPU-side).

Survey §1.2 ADOPT: every RLVR pilot must beat RFT on the same
verifier, or the RL machinery is unjustified. Samples N completions
per GRPO pilot prompt from a served checkpoint (SafeTune stage
output), keeps verifier-passing ones (``hsea.rlvr.verifier_passes``),
and emits SFT rows for a short filtered-SFT run (via
``infra/modal/safetune_qwen38.py`` with dataset/parent/out overrides).

Commands (run from the repo root, needs a served checkpoint URL)::

    .venv/bin/python scripts/issue8_rft.py sample --url <serve/v1> [--n 8 --limit 200]
    .venv/bin/python scripts/issue8_rft.py filter
    .venv/bin/python scripts/issue8_rft.py report

Reads ``runs/issue-8/grpo_prompts.jsonl``; writes ``runs/issue-8/``
(rft_samples.jsonl, rft_filtered.jsonl, rft_report.json).
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

from hsea.rlvr.rewards import combined_reward, verifier_passes  # noqa: E402

RUN_DIR = ROOT / "runs" / "issue-8"
PROMPTS = RUN_DIR / "grpo_prompts.jsonl"
SAMPLES = RUN_DIR / "rft_samples.jsonl"
FILTERED = RUN_DIR / "rft_filtered.jsonl"
REPORT = RUN_DIR / "rft_report.json"

SEED = 8
DEFAULT_N = 8
DEFAULT_LIMIT = 200


def _client(base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=os.environ.get("MODAL_API_KEY", "dummy"),
                  base_url=base_url)


def cmd_sample(url: str, model: str, n: int, limit: int,
               temperature: float = 1.0, workers: int = 1,
               no_think: bool = False, offset: int = 0) -> None:
    """Sample N completions per prompt from the served checkpoint.

    ``workers`` parallelizes the requests (server bills by wall time,
    so concurrency cuts GPU cost ~linearly to ~8 workers).
    ``no_think`` passes ``enable_thinking=False`` through vLLM's
    ``chat_template_kwargs``: Qwen3.8's spoken reasoning (~200-600
    tokens) otherwise fills the 256-token budget and the verifier
    sees CoT-only truncations (measured 4.4% pass with thinking).
    """
    import concurrent.futures  # noqa: PLC0415

    prompts = [json.loads(line) for line in open(PROMPTS)]
    prompts = prompts[offset:offset + limit] if offset else prompts[:limit]
    client = _client(url)
    t0 = time.time()
    jobs = [(rec, k) for rec in prompts for k in range(n)]
    extra = ({"chat_template_kwargs": {"enable_thinking": False}}
             if no_think else None)

    def _one(job):
        rec, k = job
        kwargs: dict = {"model": model, "temperature": temperature,
                        "max_tokens": 256,
                        "messages": [
                            {"role": "system", "content": rec["system"]},
                            {"role": "user", "content": rec["prompt"]},
                        ]}
        if extra is not None:
            kwargs["extra_body"] = extra
        resp = client.chat.completions.create(**kwargs)
        return {"id": rec["id"], "k": k,
                "completion": resp.choices[0].message.content or "",
                "gold": rec["gold"]}

    total = 0
    with open(SAMPLES, "w") as fh:
        if workers <= 1:
            for job in jobs:
                fh.write(json.dumps(_one(job)) + "\n")
                total += 1
                if total % 50 == 0:
                    print(f"sampled {total}/{len(jobs)}", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers) as pool:
                for row in pool.map(_one, jobs):
                    fh.write(json.dumps(row) + "\n")
                    total += 1
                    if total % 50 == 0:
                        print(f"sampled {total}/{len(jobs)}", flush=True)
    minutes = (time.time() - t0) / 60
    print(f"samples={total} minutes={minutes:.1f} -> {SAMPLES}")


def filter_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    """Verifier-filter (pure): keep passing completions as SFT rows."""
    prompts = {json.loads(line)["id"]: json.loads(line)
               for line in open(PROMPTS)}
    kept: list[dict] = []
    per_prompt: dict[str, dict] = {}
    for rec in samples:
        src = prompts.get(rec["id"])
        if src is None:
            continue
        ok = verifier_passes(rec["completion"], rec["gold"])
        slot = per_prompt.setdefault(rec["id"], {"n": 0, "pass": 0})
        slot["n"] += 1
        slot["pass"] += int(ok)
        if ok:
            kept.append({
                "messages": [
                    {"role": "system", "content": src["system"]},
                    {"role": "user", "content": src["prompt"]},
                    {"role": "assistant", "content": rec["completion"]},
                ],
                "source": "rft:filtered",
            })
    n = len(per_prompt)
    report = {
        "prompts": n,
        "samples": len(samples),
        "kept": len(kept),
        "verifier_pass_rate": (sum(s["pass"] for s in per_prompt.values())
                               / max(1, len(samples))),
        "prompts_with_pass": sum(1 for s in per_prompt.values()
                                 if s["pass"] > 0),
    }
    return kept, report


def cmd_filter() -> None:
    """Filter sampled completions into SFT rows + report."""
    samples = [json.loads(line) for line in open(SAMPLES)]
    kept, report = filter_samples(samples)
    with open(FILTERED, "w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_report() -> None:
    """Print the RFT report (verifier numbers the GRPO pilot must beat)."""
    print(json.dumps(json.loads(REPORT.read_text()), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample", help="sample N completions per prompt")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-think", action="store_true",
                   help="disable spoken reasoning via template kwargs")
    p.add_argument("--offset", type=int, default=0,
                   help="prompt offset (held-out reads skip train slice)")
    sub.add_parser("filter", help="verifier-filter into SFT rows")
    sub.add_parser("report", help="print the RFT report")
    args = parser.parse_args(argv)
    if args.cmd == "sample":
        cmd_sample(args.url, args.model, args.n, args.limit,
                   args.temperature, args.workers, args.no_think,
                   args.offset)
    elif args.cmd == "filter":
        cmd_filter()
    elif args.cmd == "report":
        cmd_report()


if __name__ == "__main__":
    main()
