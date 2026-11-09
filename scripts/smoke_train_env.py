#!/usr/bin/env python3
"""Smoke test for the pinned Python 3.12 Lexsi training env (issue #26).

Run ONLY inside the training env (default .venv is Python 3.10)::

    .venv-train/bin/python scripts/smoke_train_env.py

Checks: all 6 Lexsi libs import in one process with exact pinned versions,
plus one AGPL-safe API touch per lib (AlignTune TRL backend, CuratorKIT
non-PDF JSONL ingest). Exits non-zero on any failure.

AgentTune Known Issues workarounds baked in (see docs/runbooks.md):
uses GraphRunner (not StageWiseRunner) and CollectRunner (not
AuditReader.extract_dpo_pairs) for branching/DPO paths, avoids ToolRegistry
(langchain_community-dependent), and never trusts DistilledJudge/Rouge/BLEU
scores without their optional deps present.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Sandboxed shells deny ~/.cache writes (same reason uv needs UV_CACHE_DIR):
# point XDG caches at the temp dir unless the caller already set one.
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "huggingface"))

PINS = {
    "aligntune": "0.1.32",
    "curatorkit": "0.1.23",
    "safetune": "0.1.13",
    "agenttune": "1.1.3",
    "auditkit": "1.1.6",
    "circuitkit": "0.2.1",
}

FAILURES: list[str] = []


def check(name, fn):
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - smoke test reports, not raises
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"[ok] {name} {detail}")


def check_python():
    assert sys.version_info >= (3, 12), f"need Python >=3.12, got {sys.version}"
    return f"(python {sys.version.split()[0]})"


def check_versions():
    import aligntune
    import curatorkit
    import safetune
    import agenttune
    import auditkit
    import circuitkit

    mods = {
        "aligntune": aligntune,
        "curatorkit": curatorkit,
        "safetune": safetune,
        "agenttune": agenttune,
        "auditkit": auditkit,
        "circuitkit": circuitkit,
    }
    got = {}
    for name, mod in mods.items():
        ver = getattr(mod, "__version__", None) or getattr(mod, "version", None)
        if callable(ver):
            ver = ver()
        got[name] = str(ver)
    assert got == PINS, f"version drift: {got} != {PINS}"
    return "(" + ", ".join(f"{k} {v}" for k, v in got.items()) + ")"


def check_aligntune_trl():
    # AGPL guard: the AGPL PyPI `unsloth` distribution must NOT be installed
    # (only the `unsloth` extra pulls it; base aligntune does not depend on it).
    # NOTE: do NOT `import unsloth` here — the name resolves to a vendor shim
    # bundled with aligntune/agenttune whose import hard-crashes (nanobind/MLX)
    # on macOS ARM. AlignTune itself reports unsloth unavailable (see below).
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("unsloth")
    except PackageNotFoundError:
        v = None
    assert v is None, f"AGPL unsloth dist installed ({v}) — TRL-only env required"
    from aligntune import BackendFactory

    avail = BackendFactory.list_available_backends()
    assert "trl" in avail.get("SFT", []), f"TRL SFT backend missing: {avail}"
    assert "trl" in avail.get("RL", {}).get("dpo", []), f"TRL DPO missing: {avail}"
    return "(TRL SFT+DPO available, unsloth dist absent)"


def check_curator_jsonl():
    # AGPL guard: MinerU (curatorkit[pdf]) must NOT be installed.
    # find_spec avoids executing a heavy optional import; absence of the
    # importable name implies absence of the AGPL PDF path.
    from importlib.util import find_spec

    assert find_spec("mineru") is None, "mineru installed (AGPL-3.0) — non-PDF only"
    from curatorkit import JSONLReader

    rows = [
        {"prompt": f"dispatch flood zone {i}", "response": f"evacuate sector {i}"}
        for i in range(5)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        reader = JSONLReader(str(path))
        samples, rejected = reader.read()  # gated curation: (accepted, rejected)
    total = len(samples) + len(rejected)
    assert total == 5, f"expected 5 rows total, got {total}"
    return f"(5-row JSONL ingest ok: {len(samples)} accepted, {len(rejected)} gated)"


def check_safetune():
    # safetune.evaluate / safetune.harden are submodules; the entry points
    # live inside them (evaluate.evaluate fn, harden.*Trainer classes).
    from safetune import evaluate, harden

    assert callable(evaluate.evaluate), "safetune.evaluate.evaluate missing"
    assert hasattr(harden, "AsFTTrainer"), "safetune.harden trainers missing"
    return "(evaluate.evaluate fn + harden trainers present)"


def check_agenttune():
    # Known Issues: EventLog lives at agenttune.agentic.events (not top level);
    # GraphRunner for branching (StageWiseRunner ignores routing);
    # CollectRunner for DPO/BCO records (AuditReader.extract_dpo_pairs empty).
    from agenttune.agentic.events import EventLog
    from agenttune.decide.collect_runner import CollectRunner
    from agenttune.decide.graph_runner import GraphRunner

    log = EventLog()
    assert len(log.events) == 0
    assert GraphRunner is not None and CollectRunner is not None
    return "(EventLog+GraphRunner+CollectRunner ok, ToolRegistry avoided)"


def check_auditkit():
    import auditkit
    from auditkit import Sample

    samples = [
        Sample(input="flooded road", target="flooded road"),
        Sample(input="clear road", target="clear road"),
    ]
    result = auditkit.evaluate(dataset=samples, model=lambda xs: list(xs))
    assert result is not None
    n = len(getattr(result, "results", None) or samples)
    return f"(toy eval ok, {n} samples scored)"


def check_circuitkit():
    # Import-order note (2026-11-09): both circuitkit-first and
    # circuitkit-last orders verified OK with transformer-lens 3.8.0 +
    # transformers 5.22.2; no constrained order needed in this env.
    import circuitkit

    assert circuitkit.__version__ == PINS["circuitkit"]
    return f"(v{circuitkit.__version__}, order unconstrained)"


def main():
    print("== train env smoke (issue #26) ==")
    check("python>=3.12", check_python)
    check("versions", check_versions)
    check("aligntune-trl", check_aligntune_trl)
    check("curatorkit-jsonl", check_curator_jsonl)
    check("safetune", check_safetune)
    check("agenttune", check_agenttune)
    check("auditkit", check_auditkit)
    check("circuitkit", check_circuitkit)
    if FAILURES:
        print(f"\nSMOKE FAILED ({len(FAILURES)} failures)")
        raise SystemExit(1)
    print("\nSMOKE PASSED (6/6 libs, pins exact)")


if __name__ == "__main__":
    main()
