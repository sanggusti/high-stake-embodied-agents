"""L3 capture paths (#16): backend-native replay artifacts per sample.

Layout: ``runs/<run_id>/sim/<sample_id>/`` — a sibling of the L2
``trajectories/`` dir. When ``DMB_WATCH_DIR`` points at a ``trajectories/``
dir (the standard runner setup), the sim dir is resolved beside it so L1,
L2 and L3 link by run; otherwise it falls back to ``DMB_L3_DIR`` or
``runs/<run_id>/sim``. ``DMB_L3=off`` disables capture (opt-out).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def l3_enabled() -> bool:
    return os.environ.get("DMB_L3", "").lower() != "off"


def l3_root() -> Path:
    override = os.environ.get("DMB_L3_DIR")
    if override:
        return Path(override)
    watch = os.environ.get("DMB_WATCH_DIR")
    if watch:
        watch_path = Path(watch)
        if watch_path.name == "trajectories" and watch_path.parent.name:
            return watch_path.parent / "sim"
        return watch_path / "sim"
    run_id = os.environ.get("DMB_RUN_ID") or (
        "watch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    return Path("runs") / run_id / "sim"


def l3_sample_dir(sample_id: str) -> Path:
    path = l3_root() / sample_id
    path.mkdir(parents=True, exist_ok=True)
    return path
