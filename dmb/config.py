"""Environment-driven settings for Disaster Mitigation Bench."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _getenv(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """All runtime knobs. Load from environment; see `.env.example`."""

    baseten_api_key: str = ""
    baseten_base_url: str = "https://inference.baseten.co/v1"
    modal_api_key: str = ""
    modal_base_url: str = ""
    sim_backend: str = "mock"
    scenarios: tuple[str, ...] = ("all",)
    runs_dir: str = "runs"
    edge_report_interval_steps: int = 5
    edge_max_snapshots_per_episode: int = 8
    edge_perception: str = "stub"
    sam_fuse_stub: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        scenarios = tuple(
            s.strip() for s in _getenv("DMB_SCENARIOS", "all").split(",") if s.strip()
        )
        return cls(
            baseten_api_key=_getenv("BASETEN_API_KEY"),
            baseten_base_url=_getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
            modal_api_key=_getenv("MODAL_API_KEY"),
            modal_base_url=_getenv("MODAL_BASE_URL"),
            sim_backend=_getenv("DMB_SIM_BACKEND", "mock"),
            scenarios=scenarios or ("all",),
            runs_dir=_getenv("DMB_RUNS_DIR", "runs"),
            edge_report_interval_steps=int(_getenv("DMB_EDGE_REPORT_INTERVAL_STEPS", "5")),
            edge_max_snapshots_per_episode=int(
                _getenv("DMB_EDGE_MAX_SNAPSHOTS_PER_EPISODE", "8")
            ),
            edge_perception=_getenv("DMB_EDGE_PERCEPTION", "stub").lower() or "stub",
            sam_fuse_stub=_getenv("DMB_SAM_FUSE_STUB", "0") == "1",
        )

    def require_provider(self, service: str) -> None:
        """Raise a helpful error if a provider's credentials are missing."""
        if service == "baseten" and not self.baseten_api_key:
            raise RuntimeError(
                "BASETEN_API_KEY is not set. Export it or copy .env.example to .env."
            )
        if service == "modal" and (not self.modal_api_key or not self.modal_base_url):
            raise RuntimeError(
                "MODAL_API_KEY / MODAL_BASE_URL are not set. Deploy "
                "infra/modal/vllm_server.py first, then set both."
            )
