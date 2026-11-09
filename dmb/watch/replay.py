"""T1 event replay: narrate an L2 JSONL log as a timeline.

``python -m dmb.watch.replay <sample.jsonl> [--filter action,edge]``
Minimal data-QC tool for Phase D (partial #16); T2/T3 stay in #16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def narrate(path: str | Path, kinds: set[str] | None = None) -> list[str]:
    lines: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        event = json.loads(raw)
        if kinds and event.get("kind") not in kinds:
            continue
        t, step, kind = event.get("t"), event.get("step"), event.get("kind")
        head = f"[t={t:7.1f} step={step:>3}] {kind:6}"
        if kind == "action":
            args = ", ".join(f"{k}={v}" for k, v in (event.get("args") or {}).items())
            lines.append(f"{head} {event.get('tool')}({args}) -> {event.get('obs', '')[:90]}")
        elif kind == "edge":
            lines.append(f"{head} ({event.get('trigger')}) {event.get('summary', '')[:100]}")
        elif kind == "state":
            lines.append(
                f"{head} rescued={event.get('rescued')} "
                f"violations={event.get('violations')} "
                f"coverage={event.get('coverage'):.2f}"
            )
        elif kind == "score":
            lines.append(
                f"{head} composite={event.get('composite', 0):.3f} "
                f"recall={event.get('rescue_recall', 0):.2f} "
                f"contain={event.get('containment', 0):.2f} "
                f"eff={event.get('efficiency', 0):.2f} "
                f"safety={event.get('safety', 0):.2f}"
            )
        else:
            detail = event.get("event", "")
            extra = ""
            if detail == "episode_start":
                extra = f" {event.get('scenario')} seed={event.get('seed')} {event.get('backend')}"
            lines.append(f"{head} {detail}{extra}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Narrate an L2 trajectory file.")
    parser.add_argument("path")
    parser.add_argument("--filter", default="", help="comma-separated kinds")
    args = parser.parse_args()
    kinds = set(args.filter.split(",")) if args.filter else None
    for line in narrate(args.path, kinds):
        print(line)


if __name__ == "__main__":
    main()
