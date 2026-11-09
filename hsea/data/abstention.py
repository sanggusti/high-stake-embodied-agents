"""Abstention/escalation eval set (issue #21): loader, validator, manifest.

Curator-written text/tool-trace scenarios (no sim): the agent must abstain
or escalate on uncertain/high-stakes calls. A paired benign set guards
against over-refusal (AgentHarm pattern). Playbook:
``docs/alignment-research.md`` §4 (Calibration / refusal / abstention).

Schema per JSONL record::

    {"id": "abs-001", "set": "risk|benign", "split": "cal|eval",
     "category": str, "label": "abstain|escalate|comply",
     "prompt": str, "rationale": str, "stakes": "low|medium|high|critical",
     "provenance": {"curator": str, "playbook": str, "issue": 21}}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LABELS = ("abstain", "escalate", "comply")
SETS = ("risk", "benign")
SPLITS = ("cal", "eval")
STAKES = ("low", "medium", "high", "critical")

DATA_FILES = {"risk": "scenarios.jsonl", "benign": "benign.jsonl"}

REQUIRED_KEYS = (
    "id", "set", "split", "category", "label",
    "prompt", "rationale", "stakes", "provenance",
)


def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "abstention"


def validate_scenario(rec: dict) -> dict:
    """Validate one record; return it unchanged. Raises ValueError."""
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise ValueError(f"{rec.get('id', '?')}: missing keys {missing}")
    if rec["set"] not in SETS:
        raise ValueError(f"{rec['id']}: bad set {rec['set']!r}")
    if rec["split"] not in SPLITS:
        raise ValueError(f"{rec['id']}: bad split {rec['split']!r}")
    if rec["label"] not in LABELS:
        raise ValueError(f"{rec['id']}: bad label {rec['label']!r}")
    if rec["set"] == "benign" and rec["label"] != "comply":
        raise ValueError(f"{rec['id']}: benign set must be comply-only")
    if rec["stakes"] not in STAKES:
        raise ValueError(f"{rec['id']}: bad stakes {rec['stakes']!r}")
    for key in ("prompt", "rationale", "category"):
        if not isinstance(rec[key], str) or not rec[key].strip():
            raise ValueError(f"{rec['id']}: empty {key}")
    prov = rec["provenance"]
    if not isinstance(prov, dict) or prov.get("issue") != 21:
        raise ValueError(f"{rec['id']}: provenance must name issue 21")
    return rec


def load_scenarios(
    root: str | Path | None = None, *, sets: tuple[str, ...] = SETS
) -> list[dict]:
    """Load + validate records from ``data/abstention/*.jsonl``."""
    root = Path(root) if root is not None else data_dir()
    records = []
    for set_name in sets:
        path = root / DATA_FILES[set_name]
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: bad JSON ({exc})") from exc
            if rec.get("set") != set_name:
                raise ValueError(
                    f"{path.name}:{lineno}: set field {rec.get('set')!r} "
                    f"!= file set {set_name!r}"
                )
            records.append(validate_scenario(rec))
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate ids: {dupes}")
    return records


def record_hash(rec: dict) -> str:
    """SHA-256 over the canonical (sorted-keys) record JSON."""
    blob = json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_manifest(records: list[dict], root: str | Path) -> dict:
    """Manifest: counts, label/split distribution, SHA-256 (file + record)."""
    root = Path(root)
    manifest: dict = {
        "issue": 21,
        "provenance": {
            "curator": "repo curator (issue #21)",
            "playbook": "docs/alignment-research.md §4 Calibration/refusal/abstention",
            "benign_pattern": "AgentHarm paired benign set",
        },
        "counts": {},
        "label_counts": {},
        "split_counts": {},
        "files": {},
        "record_hashes": {},
    }
    for rec in records:
        key = rec["set"]
        manifest["counts"][key] = manifest["counts"].get(key, 0) + 1
        label_key = f"{rec['set']}.{rec['label']}"
        manifest["label_counts"][label_key] = manifest["label_counts"].get(label_key, 0) + 1
        split_key = f"{rec['set']}.{rec['split']}"
        manifest["split_counts"][split_key] = manifest["split_counts"].get(split_key, 0) + 1
        manifest["record_hashes"][rec["id"]] = record_hash(rec)
    for set_name, fname in DATA_FILES.items():
        blob = (root / fname).read_bytes()
        manifest["files"][fname] = hashlib.sha256(blob).hexdigest()
    manifest["records_total"] = len(records)
    return manifest


def write_manifest(root: str | Path | None = None) -> dict:
    root = Path(root) if root is not None else data_dir()
    records = load_scenarios(root)
    manifest = build_manifest(records, root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(root: str | Path | None = None) -> dict:
    """Re-hash files + records; return mismatches (empty dict == clean)."""
    root = Path(root) if root is not None else data_dir()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    problems: dict = {"files": {}, "records": {}, "missing": []}
    for fname, want in manifest.get("files", {}).items():
        path = root / fname
        if not path.exists():
            problems["missing"].append(fname)
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            problems["files"][fname] = {"want": want, "got": got}
    records = {r["id"]: r for r in load_scenarios(root)}
    for rid, want in manifest.get("record_hashes", {}).items():
        rec = records.get(rid)
        if rec is None:
            problems["missing"].append(rid)
        elif record_hash(rec) != want:
            problems["records"][rid] = {"want": want, "got": record_hash(rec)}
    return {k: v for k, v in problems.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Abstention set manifest tools.")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    if args.verify_only:
        problems = verify_manifest(args.root)
        print(json.dumps(problems, indent=2))
        raise SystemExit(1 if problems else 0)
    manifest = write_manifest(args.root)
    print(json.dumps({k: manifest[k] for k in ("records_total", "counts")}, indent=2))


if __name__ == "__main__":
    main()
