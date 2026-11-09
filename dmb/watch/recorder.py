"""L2 event recorder: timestamped sim events, one JSONL file per sample.

Contract (docs/watcher.md): ``trajectories/<sample_id>.jsonl``, one event
per line, ``kind`` in action|edge|state|score|note. ``t`` is sim seconds
(monotonic); backends map wall time, mock uses its nominal 5 s/step clock.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class Watcher(Protocol):
    def on_start(self, metadata: dict) -> None: ...
    def on_event(self, kind: str, t: float, step: int, **fields: object) -> None: ...
    def on_end(self, summary: dict) -> None: ...


class NullWatcher:
    """No-op watcher for tests and disabled runs."""

    def on_start(self, metadata: dict) -> None:
        return None

    def on_event(self, kind: str, t: float, step: int, **fields: object) -> None:
        return None

    def on_end(self, summary: dict) -> None:
        return None


def _run_dir() -> Path:
    base = os.environ.get("DMB_WATCH_DIR")
    if base:
        return Path(base)
    run_id = os.environ.get("DMB_RUN_ID") or (
        "watch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    return Path("runs") / run_id / "trajectories"


class JsonlWatcher:
    """Append-only JSONL watcher rooted at the run trajectories dir."""

    def __init__(self, sample_id: str, run_dir: Path | None = None) -> None:
        self.sample_id = sample_id
        self.run_dir = Path(run_dir) if run_dir else _run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / f"{sample_id}.jsonl"
        # Fresh file per episode: a retry overwrites, never interleaves.
        self.path.write_text("", encoding="utf-8")

    def on_start(self, metadata: dict) -> None:
        self.on_event(
            "note",
            0.0,
            0,
            event="episode_start",
            scenario=metadata.get("scenario"),
            seed=metadata.get("seed"),
            world_hash=metadata.get("world_hash"),
            backend=metadata.get("backend"),
        )

    def on_event(self, kind: str, t: float, step: int, **fields: object) -> None:
        line = json.dumps({"t": t, "step": step, "kind": kind, **fields})
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def on_end(self, summary: dict) -> None:
        state = summary.get("state", {})
        if isinstance(state, dict):
            self.on_event(
                "state",
                float(state.get("sim_time", 0.0)),
                int(state.get("steps", 0)),
                rescued=sum(1 for v in state.get("victims", []) if v.get("found")),
                violations=int(state.get("safety_violations", 0)),
                coverage=float(state.get("coverage", 0.0)),
            )
        self.on_event(
            "note",
            float(state.get("sim_time", 0.0)) if isinstance(state, dict) else 0.0,
            int(state.get("steps", 0)) if isinstance(state, dict) else 0,
            event="episode_end",
        )


def watcher_from_env(sample_id: str) -> Watcher:
    """Build the run watcher; ``DMB_WATCH=off`` disables recording."""
    if os.environ.get("DMB_WATCH", "").lower() == "off":
        return NullWatcher()
    return JsonlWatcher(sample_id)


def record_score(sample_id: str, breakdown: dict) -> None:
    """Append the scorer's metric vector (audit rule: scores from L2)."""
    if os.environ.get("DMB_WATCH", "").lower() == "off":
        return
    path = _run_dir() / f"{sample_id}.jsonl"
    if not path.exists():
        return
    line = json.dumps({"t": -1.0, "step": -1, "kind": "score", **breakdown})
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
