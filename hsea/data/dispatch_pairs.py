"""Dispatch-priority DPO pairs from crisis-to-action labels (issue #20).

Converts TREC-IS criticality + HumAID/CrisisMMD humanitarian labels into
``(situation, better/worse dispatch)`` preference pairs for flywheel-v2.
Situation text is derived from labels only — no raw tweet text is read,
stored, or written, per Twitter redistribution constraints (see
``data/dispatch-pairs/LICENSE_NOTES.md``).

Input: one JSONL file per source; each line is a record::

    {"id": "<source record id>", "labels": {...}, "seed": <optional int>}

Label schemas per source:

- ``trec-is``: ``{"info_types": [...], "criticality": "critical|high|medium|low"}``
- ``humaid``: ``{"humanitarian": "<category>"}``
- ``crisismmd``: ``{"informative": bool, "humanitarian": "<category>",
  "severity": "severe|mild|none" (optional)}``

Output, under ``out_dir``: ``dpo.jsonl`` (one pair per line) + ``manifest.json``
(code commit, per-source counts, file SHA-256, per-pair hashes). Mirrors
``hsea.data.export`` manifest conventions for #7 curation.

``python -m hsea.data.dispatch_pairs --trec-is trec.jsonl --humaid hum.jsonl
--crisismmd cmmd.jsonl --out data/dispatch-pairs --split train``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hsea.data.splits import split_for_seed

SOURCES = ("trec-is", "humaid", "crisismmd")

URGENT_HUMAID = frozenset(
    {
        "injured_or_dead_people",
        "missing_or_found_people",
        "requests_or_urgent_needs",
        "displaced_people_and_evacuations",
    }
)
ROUTINE_HUMAID = frozenset(
    {
        "infrastructure_and_utility_damage",
        "rescue_volunteering_or_donation_effort",
        "caution_and_advice",
        "other_relevant_information",
    }
)
URGENT_INFO_HINTS = ("searchandrescue", "emergingthreats", "movepeople", "firstpartyobservation")


def _norm(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def priority_for(source: str, labels: dict) -> tuple[int, str]:
    """Map source labels to a dispatch priority (1/2/3) plus a reason string."""
    if source == "trec-is":
        crit = _norm(labels.get("criticality"))
        if crit not in ("critical", "high", "medium", "low"):
            raise ValueError(f"trec-is record needs criticality, got {crit!r}")
        types = [str(t) for t in labels.get("info_types", []) or []]
        urgent_type = any(h in _norm(t) for t in types for h in URGENT_INFO_HINTS)
        detail = ", ".join(types) if types else "no info type"
        if crit in ("critical", "high"):
            return 1, f"TREC-IS criticality is {crit} ({detail})"
        if crit == "medium":
            if urgent_type:
                return 1, f"TREC-IS medium criticality but urgent type ({detail})"
            return 2, f"TREC-IS criticality is medium ({detail})"
        return 3, f"TREC-IS criticality is low ({detail})"
    if source == "humaid":
        cat = _norm(labels.get("humanitarian"))
        if not cat:
            raise ValueError("humaid record needs a humanitarian category")
        if cat in URGENT_HUMAID:
            return 1, f"HumAID category {cat} needs urgent response"
        if cat in ROUTINE_HUMAID:
            return 2, f"HumAID category {cat} needs routine follow-up"
        if cat in ("not_humanitarian", "sympathy_and_support"):
            return 3, f"HumAID category {cat} needs no dispatch"
        return 2, f"HumAID category {cat} unmapped, default routine review"
    if source == "crisismmd":
        if not labels.get("informative"):
            return 3, "CrisisMMD report is not informative"
        sev = _norm(labels.get("severity"))
        cat = _norm(labels.get("humanitarian")) or "unknown"
        if sev == "severe":
            return 1, f"CrisisMMD severe damage ({cat})"
        if sev == "mild":
            return 2, f"CrisisMMD mild damage ({cat})"
        if sev == "none":
            return 3, f"CrisisMMD no damage ({cat})"
        if sev:
            raise ValueError(f"crisismmd record needs severity, got {sev!r}")
        if cat in URGENT_HUMAID:
            return 1, f"CrisisMMD severity unstated but {cat} needs urgent response"
        if cat in ROUTINE_HUMAID:
            return 2, f"CrisisMMD severity unstated; {cat} needs routine follow-up"
        if cat in ("not_humanitarian", "sympathy_and_support"):
            return 3, f"CrisisMMD severity unstated; {cat} needs no dispatch"
        return 2, f"CrisisMMD severity unstated; {cat} unmapped, default routine review"
    raise ValueError(f"unknown source {source!r}")


def situation_from_labels(source: str, record_id: str, labels: dict) -> str:
    """Label-derived situation text. Never embeds raw post text."""
    if source == "trec-is":
        types = [str(t) for t in labels.get("info_types", []) or []]
        body = (
            f"information types: {', '.join(types) if types else 'none listed'}; "
            f"assessed criticality: {_norm(labels.get('criticality'))}"
        )
    elif source == "humaid":
        body = f"humanitarian category: {_norm(labels.get('humanitarian'))}"
    elif source == "crisismmd":
        body = (
            f"informative: {'yes' if labels.get('informative') else 'no'}; "
            f"humanitarian category: {_norm(labels.get('humanitarian')) or 'unknown'}; "
            f"damage severity: {_norm(labels.get('severity')) or 'unstated'}"
        )
    else:
        raise ValueError(f"unknown source {source!r}")
    return (
        f"Crisis report ({source} record {record_id}): {body}.\n"
        "Task: choose the dispatch priority "
        "(P1 immediate / P2 routine / P3 monitor) and state the next action."
    )


def _dispatch(level: int, reason: str, *, worse: bool) -> str:
    if not worse:
        if level == 1:
            return (
                f"PRIORITY 1 — IMMEDIATE DISPATCH. {reason}. "
                "Task the nearest available unit now, notify incident command, "
                "and confirm en route within minutes."
            )
        if level == 2:
            return (
                f"PRIORITY 2 — ROUTINE DISPATCH. {reason}. "
                "Queue for the next available unit and confirm completion "
                "within the operational period."
            )
        return (
            f"PRIORITY 3 — MONITOR. {reason}. "
            "Log, no unit tasked; re-evaluate only if new information arrives."
        )
    if level == 3:
        return (
            "PRIORITY 1 — IMMEDIATE DISPATCH. Scramble the nearest unit with "
            "lights and sirens and pull incident command off other tasks."
        )
    return (
        "PRIORITY 3 — MONITOR. Log this report and revisit it in the next "
        "routine cycle; no unit tasked, no escalation."
    )


def _provenance_labels(source: str, labels: dict) -> dict:
    keep = {
        "trec-is": ("info_types", "criticality"),
        "humaid": ("humanitarian",),
        "crisismmd": ("informative", "humanitarian", "severity"),
    }[source]
    return {k: labels[k] for k in keep if k in labels}


def make_pair(source: str, record_id: str, labels: dict, split: str) -> dict:
    """Build one DPO pair from a single labeled record."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be train/test, got {split!r}")
    level, reason = priority_for(source, labels)
    return {
        "situation": situation_from_labels(source, record_id, labels),
        "chosen": _dispatch(level, reason, worse=False),
        "rejected": _dispatch(level, reason, worse=True),
        "priority": f"P{level}",
        "split": split,
        "provenance": {
            "source": source,
            "record_id": record_id,
            "labels": _provenance_labels(source, labels),
        },
    }


def validate_pair(pair: dict) -> None:
    """Raise ValueError unless the pair matches the DPO schema."""
    for key in ("situation", "chosen", "rejected", "priority", "split", "provenance"):
        if key not in pair:
            raise ValueError(f"pair missing key {key!r}")
    if pair["priority"] not in ("P1", "P2", "P3"):
        raise ValueError(f"bad priority {pair['priority']!r}")
    if pair["split"] not in ("train", "test"):
        raise ValueError(f"bad split {pair['split']!r}")
    if not pair["situation"] or pair["chosen"] == pair["rejected"]:
        raise ValueError("pair needs a situation and distinct chosen/rejected")
    prov = pair["provenance"]
    if prov.get("source") not in SOURCES or not prov.get("record_id"):
        raise ValueError(f"bad provenance {prov!r}")


def read_input(path: str | Path, source: str) -> list[dict]:
    """Read a source JSONL file; each line needs ``id`` + ``labels``."""
    records = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: bad JSON: {exc}") from exc
        if not obj.get("id") or not isinstance(obj.get("labels"), dict):
            raise ValueError(f"{path}:{lineno}: record needs id + labels dict")
        records.append(obj)
    return records


def build_pairs(
    sources: dict[str, list[dict]],
    split: str,
    *,
    max_pairs: int | None = None,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Build pairs from per-source records; returns (pairs, counts)."""
    pairs: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for source, records in sources.items():
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        made = 0
        for rec in records:
            if "seed" in rec and split_for_seed(int(rec["seed"])) != split:
                raise ValueError(
                    f"{source} record {rec['id']} seed {rec['seed']} "
                    f"is not in split {split!r}"
                )
            pairs.append(make_pair(source, str(rec["id"]), rec["labels"], split))
            made += 1
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
        counts[source] = {"records": len(records), "pairs": made}
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    for source in sources:
        counts.setdefault(source, {"records": 0, "pairs": 0})
    return pairs, counts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_export(
    pairs: list[dict],
    counts: dict[str, dict[str, int]],
    out_dir: str | Path,
    split: str,
) -> dict:
    """Write ``dpo.jsonl`` + ``manifest.json``; returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dpo_path = out / "dpo.jsonl"
    with dpo_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            validate_pair(pair)
            fh.write(json.dumps(pair, sort_keys=True) + "\n")
    priority_counts = {"P1": 0, "P2": 0, "P3": 0}
    pair_hashes = {}
    for pair in pairs:
        priority_counts[pair["priority"]] += 1
        key = f"{pair['provenance']['source']}:{pair['provenance']['record_id']}"
        pair_hashes[key] = hashlib.sha256(
            json.dumps(pair, sort_keys=True).encode()
        ).hexdigest()
    manifest = {
        "code_commit": _code_commit(),
        "split": split,
        "sources": counts,
        "pairs_total": len(pairs),
        "priority_counts": priority_counts,
        "files": {"dpo.jsonl": _sha256(dpo_path)},
        "pair_hashes": pair_hashes,
        "license_notes": "see LICENSE_NOTES.md; NC sources are research-only (#24 gates v2)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_export(out_dir: str | Path) -> dict:
    """Re-verify an export: file hashes, pair hashes, schema. Returns summary."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    dpo_path = out / "dpo.jsonl"
    actual = _sha256(dpo_path)
    if manifest["files"]["dpo.jsonl"] != actual:
        raise ValueError(f"dpo.jsonl SHA-256 mismatch: {actual} != manifest")
    pairs = [json.loads(line) for line in dpo_path.read_text(encoding="utf-8").splitlines()]
    if len(pairs) != manifest["pairs_total"]:
        raise ValueError("pair count != manifest pairs_total")
    for pair in pairs:
        validate_pair(pair)
        key = f"{pair['provenance']['source']}:{pair['provenance']['record_id']}"
        want = manifest["pair_hashes"].get(key)
        got = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
        if want != got:
            raise ValueError(f"pair hash mismatch for {key}")
    return {"pairs": len(pairs), "sources": manifest["sources"], "split": manifest["split"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build dispatch-priority DPO pairs.")
    parser.add_argument("--trec-is", default=None, help="TREC-IS records JSONL")
    parser.add_argument("--humaid", default=None, help="HumAID records JSONL")
    parser.add_argument("--crisismmd", default=None, help="CrisisMMD records JSONL")
    parser.add_argument("--out", required=True, help="output dir for dpo.jsonl + manifest")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="parse + print counts, no writes")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing export")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            print(json.dumps(verify_export(args.out), indent=2))
            return
        inputs = {"trec-is": args.trec_is, "humaid": args.humaid, "crisismmd": args.crisismmd}
        inputs = {k: v for k, v in inputs.items() if v}
        if not inputs:
            raise ValueError("need at least one of --trec-is/--humaid/--crisismmd")
        sources = {k: read_input(v, k) for k, v in inputs.items()}
        pairs, counts = build_pairs(sources, args.split, max_pairs=args.max_pairs)
        if args.dry_run:
            print(json.dumps({"pairs": len(pairs), "sources": counts}, indent=2))
            return
        print(json.dumps(write_export(pairs, counts, args.out, args.split), indent=2))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
