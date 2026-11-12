"""Shared SafeTune repair-mix composition primitives (#10).

Consolidates the pieces the per-loop ``compose_aya7*_repair.py``
scripts prove independently: seeded stratified sampling, the KTO
binary-label contract + canonical KTO trainer defaults (mirrored
inline in the Modal training legs, which cannot import this package
remotely — parity pinned by test), the unified-footer audit, and the
repair-manifest / held-out-disjointness contract every repair mix
must satisfy.

Held-out sets (red-team attacks, dispatch pairs, XSTest lookalikes)
stay eval-only: repair mixes must be disjoint from them by prompt
text. Iterative repair tuning itself (closing Gate 2) is a follow-on;
this module tracks and expresses the repair contract, it does not
tune.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Unified footer contract: ONE footer everywhere, ``DECISION:`` with
# the 4-way vocab. Abstention demos migrate off ``ACTION:``.
FOOTER_CONTRACT = "DECISION-unified"
FOOTER_VALUES = ("refuse", "escalate", "abstain", "comply")
LEGACY_FOOTERS = ("ACTION:",)

# Canonical KTO arm defaults, as executed in both loops
# (infra/modal/aya_expanse.py::kto, infra/modal/finetune_aya73_followup.py::kto):
# cold-start from the merged DPO checkpoint + fresh LoRA, 1 epoch, LR 5e-5.
# The Modal legs mirror these values inline (remote code cannot import
# hsea); tests/test_safetune_harness.py pins parity.
KTO_DEFAULTS = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 5,
    "logging_steps": 5,
    "save_steps": 25,
    "save_total_limit": 1,
    "bf16": True,
    "beta": 0.1,
    "max_length": 2048,
}

# Repair-manifest held-out contract: these eval-only files must never
# contribute rows to a repair mix (checked by prompt text).
HELD_OUT_FILES = (
    "data/redteam/attacks.jsonl",
    "data/dispatch-pairs/dpo.jsonl",
    "data/xstest/lookalikes.jsonl",
)


def stratified_sample(rows: list[dict], n: int, seed: int,
                      key: str = "priority") -> list[dict]:
    """Proportional sample by ``key`` (P1/P2/P3), seeded.

    Shared verbatim implementation behind both loops' repair-mix
    dispatch anchors / debiased pairs: per-bucket shuffle, proportional
    quotas (min 1), largest-bucket trim to exactly ``n``, final
    shuffle. Deterministic in (rows, n, seed).
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for rec in rows:
        buckets.setdefault(rec.get(key, "?"), []).append(rec)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    total = len(rows)
    quotas = {k: max(1, round(n * len(v) / total))
              for k, v in buckets.items()}
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=quotas.get)  # type: ignore[arg-type]
        quotas[biggest] -= 1
    out = []
    for bucket_key in sorted(buckets):
        out.extend(buckets[bucket_key][:quotas[bucket_key]])
    rng.shuffle(out)
    assert len(out) == n, f"stratified {len(out)} != {n}"
    return out


def validate_kto_rows(rows: list[dict]) -> dict:
    """Validate the KTO binary-label contract; return split stats.

    Every row needs a bool ``label`` (True = desirable gold demo,
    False = wrong-label violation); both classes must be present.
    Raises ``AssertionError`` on violation.
    """
    assert rows, "kto mix is empty"
    labels = [r["label"] for r in rows]
    assert all(isinstance(v, bool) for v in labels), \
        "kto labels must be bool"
    n_des = sum(labels)
    assert 0 < n_des < len(rows), "kto needs both classes"
    return {"total": len(rows), "desirable": n_des,
            "undesirable": len(rows) - n_des}


def audit_footers(texts: list[str], strict: bool = True) -> dict:
    """Audit the unified-footer contract over demo texts.

    Returns leak counts: legacy (``ACTION:``) footers must be zero and
    every ``DECISION:`` value must be in the 4-way vocab. With
    ``strict=True`` (the bar for new mixes) violations raise
    ``AssertionError``; with ``strict=False`` they are reported in the
    result (for auditing pre-unification mixes such as aya73's).
    """
    import re

    legacy = sum(1 for t in texts if "ACTION:" in t)
    values: set[str] = set()
    for t in texts:
        for m in re.finditer(r"DECISION:\s*([A-Za-z_]+)", t):
            values.add(m.group(1))
    unknown = sorted(set(values) - set(FOOTER_VALUES))
    if strict:
        assert legacy == 0, f"{legacy} legacy ACTION: footers leaked"
        assert not unknown, f"off-vocab footer values: {unknown}"
    return {"texts": len(texts), "legacy_leaks": legacy,
            "footer_values": sorted(values),
            "unknown_values": unknown}


def _prompts_in(path: Path) -> set[str]:
    texts = set()
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            for field in ("prompt", "situation", "chosen", "situation_text"):
                val = rec.get(field)
                if isinstance(val, str) and val.strip():
                    texts.add(val.strip())
    return texts


def assert_held_out_disjoint(mix_prompts: list[str],
                             held_out_files: tuple[str, ...] = HELD_OUT_FILES,
                             root: Path | str | None = None) -> dict:
    """Assert repair-mix prompts are disjoint from eval-only files.

    ``mix_prompts`` are the user-prompt texts in the mix; every
    held-out file contributes its prompt/situation/chosen texts.
    Raises ``AssertionError`` listing overlaps. Missing held-out files
    are skipped (noted in the result) so the check stays total on
    partial checkouts.
    """
    root = Path(root) if root else Path.cwd()
    mix = {p.strip() for p in mix_prompts if p and p.strip()}
    checked, skipped, overlaps = [], [], {}
    for rel in held_out_files:
        path = root / rel
        if not path.exists():
            skipped.append(rel)
            continue
        held = _prompts_in(path)
        hit = sorted(mix & held)
        checked.append(rel)
        if hit:
            overlaps[rel] = hit
    assert not overlaps, f"repair mix overlaps held-out evals: {overlaps}"
    return {"mix_prompts": len(mix), "checked": checked,
            "skipped": skipped}


def build_repair_manifest(issue: int, stage: int, seed: int,
                          footer_contract: str, sft_stats: dict,
                          dpo_stats: dict, kto_stats: dict,
                          extra: dict | None = None) -> dict:
    """Build the standard repair-manifest payload (JSON-serializable)."""
    manifest = {
        "issue": issue,
        "stage": stage,
        "seed": seed,
        "footer_contract": footer_contract,
        "sft": dict(sft_stats),
        "dpo": dict(dpo_stats),
        "kto": dict(kto_stats),
        "kto_defaults": dict(KTO_DEFAULTS),
        "held_out": list(HELD_OUT_FILES),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_mix(out_dir: Path | str, name: str,
              rows: list[dict]) -> Path:
    """Write one ``<name>.jsonl`` mix file; return its path."""
    path = Path(out_dir) / f"{name}.jsonl"
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path
