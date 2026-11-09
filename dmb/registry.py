"""Run registry: every eval execution leaves a reproducible record.

Layout per run::

    runs/<run_id>/
      config.json    # task, model, scenario, seeds, limits, git sha
      scores.json    # metric vector per sample (+ aggregates)
      events.jsonl   # free-form run events (sim resets, errors, notes)
      eval-log.txt   # path/pointer to the inspect-ai .eval log(s)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class RunRecord:
    run_id: str
    task: str
    model: str
    scenario: str
    seeds: list[int]
    limits: dict[str, int] = field(default_factory=dict)
    started_at: str = field(default_factory=_utcnow)
    finished_at: str | None = None
    git_sha: str | None = None
    eval_logs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)


class RunRegistry:
    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def start_run(
        self,
        run_id: str,
        *,
        task: str,
        model: str,
        scenario: str,
        seeds: list[int],
        limits: dict[str, int] | None = None,
        repo_root: str | Path = ".",
    ) -> RunRecord:
        path = self._run_path(run_id)
        path.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            run_id=run_id,
            task=task,
            model=model,
            scenario=scenario,
            seeds=list(seeds),
            limits=dict(limits or {}),
            git_sha=_git_sha(Path(repo_root)),
        )
        (path / "config.json").write_text(json.dumps(asdict(record), indent=2))
        return record

    def log_event(self, run_id: str, event: dict) -> None:
        path = self._run_path(run_id) / "events.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps({"at": _utcnow(), **event}) + "\n")

    def write_scores(self, run_id: str, scores: dict) -> None:
        path = self._run_path(run_id) / "scores.json"
        path.write_text(json.dumps(scores, indent=2))

    def finish_run(self, run_id: str, *, eval_logs: list[str] | None = None) -> RunRecord:
        path = self._run_path(run_id)
        config_path = path / "config.json"
        record = RunRecord(**json.loads(config_path.read_text()))
        record.finished_at = _utcnow()
        if eval_logs:
            record.eval_logs = list(eval_logs)
        config_path.write_text(json.dumps(asdict(record), indent=2))
        return record

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self.runs_dir.iterdir() if p.is_dir())
