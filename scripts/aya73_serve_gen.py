"""Served probe completions for issue-73 text-only tracks (T1, T5).

Calls the Modal vLLM server (infra/modal/vllm_aya73.py) over its
OpenAI-compatible API for the footage probes (T1, text-only prompts
with [frame omitted], same shape as SFT rows) and the dispatch
situations (T5, model-vs-model pairs). Safety/abstention/redteam run
through inspect-ai instead (they need .eval logs + eval ids).

Checkpoints: tinyaya73-base, tinyaya73-sft, tinyaya73-dpo, each on its
own server URL. Env: MODAL_BASE_URL_BASE, MODAL_BASE_URL_SFT,
MODAL_BASE_URL_DPO (each a serve URL + /v1), MODAL_API_KEY (any ok).
A single MODAL_BASE_URL covers all three when they share one server.

Usage:
  python3 scripts/aya73_serve_gen.py          # footage + dispatch, 3 ckpts
  python3 scripts/aya73_serve_gen.py --track footage --limit 5
Writes runs/aya73-serve/{ckpt}_{track}.jsonl with {id, completion}.
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

CHECKPOINTS = ("tinyaya73-base", "tinyaya73-sft", "tinyaya73-dpo")
RUN_DIR = ROOT / "runs" / "aya73-serve"


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if not value and (ROOT / ".env").exists():
        for line in open(ROOT / ".env"):
            line = line.strip()
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip("'\"")
    return value or None


def _base_url(ckpt: str) -> str:
    suffix = ckpt.removeprefix("tinyaya73-").upper()
    url = _env(f"MODAL_BASE_URL_{suffix}") or _env("MODAL_BASE_URL")
    if not url:
        raise SystemExit(f"missing MODAL_BASE_URL_{suffix} (env or .env)")
    return url.rstrip("/")


def _client(ckpt: str):
    from openai import OpenAI

    key = os.environ.get("MODAL_API_KEY", "modal")
    return OpenAI(api_key=key, base_url=_base_url(ckpt))


def _messages(track: str, row: dict) -> list[dict]:
    if track == "footage":
        return [{"role": "user", "content": row["user_text"]}]
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["situation"]},
    ]


def run_track(track: str, limit: int, max_retries: int = 4) -> dict[str, int]:
    clients = {ckpt: _client(ckpt) for ckpt in CHECKPOINTS}
    src = ROOT / "data" / "aya73" / (
        "eval_footage.jsonl" if track == "footage"
        else "eval_dispatch.jsonl")
    rows = [json.loads(line) for line in open(src) if line.strip()]
    if limit > 0:
        rows = rows[:limit]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for ckpt in CHECKPOINTS:
        client = clients[ckpt]
        out_rows = []
        for row in rows:
            last: Exception | None = None
            text = ""
            for attempt in range(max_retries):
                try:
                    resp = client.chat.completions.create(
                        model=ckpt, temperature=0.0, max_tokens=256,
                        messages=_messages(track, row))
                    text = resp.choices[0].message.content or ""
                    break
                except Exception as exc:  # noqa: BLE001, retry then keep ""
                    last = exc
                    time.sleep(2.0 * (attempt + 1))
            if last is not None and not text:
                print(f"WARN {ckpt} {row['id']}: {type(last).__name__}",
                      flush=True)
            out_rows.append({"id": row["id"], "completion": text})
        dest = RUN_DIR / f"{ckpt}_{track}.jsonl"
        with open(dest, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\n")
        counts[ckpt] = len(out_rows)
        print(f"wrote {dest} ({len(out_rows)})", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="all",
                        choices=("all", "footage", "dispatch"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tracks = ("footage", "dispatch") if args.track == "all" \
        else (args.track,)
    for track in tracks:
        run_track(track, args.limit)


if __name__ == "__main__":
    main()
