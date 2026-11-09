"""Curate flywheel-v2 SFT/RLVR (+ dispatch DPO) from pinned sources (#7).

Sources (all train-side, all pinned revisions; clearance asserted via
:mod:`hsea.data.clearance` before any streaming starts):

- ``floodnet-vqa`` — ``ljx620/FloodNet-VQA`` train: QA text + image refs
  (no pixels redistributed).
- ``ladi-v2`` — ``MITLL/LADI-v2-dataset`` train: expert labels rendered as
  12 yes/no QA pairs per image (same questions as #19), refs only.
- ``disasterm3`` — ``DisasterM3_Instruct/train_release.json`` instruction
  pairs (referring-seg rows excluded: mask outputs need vision SFT).
- ``crisitext`` — ``LanD-FBK/crisitext`` train: bulletin + FEMA guideline
  names + event -> warning message (synthetic ``generated`` rows excluded).
- ``humaid`` / ``crisismmd`` — labels-only dispatch DPO pairs through the
  #20 builder (no tweet text/images); TREC-IS skipped per #24.
- ``l2`` — bench episodes: v1 carry-forward (``data/flywheel-v1``) + fresh
  exports from ``logs/*.eval`` (SFT success-filtered, non-mock teachers;
  RLVR all scored episodes incl. failures as negatives).

Output, under ``out_dir``: ``sft.jsonl`` + ``dpo.jsonl`` + ``rlvr.jsonl``
+ ``manifest.json`` (counts/splits/filters per source, revs, SHA-256).

``python -m hsea.data.curate_v2 --out data/flywheel-v2 --split train``
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hsea.data.clearance import CLEARANCE, assert_cleared_for_training
from hsea.data.dispatch_pairs import build_pairs, validate_pair
from hsea.data.export import messages_to_sft_pairs
from hsea.data.footage import (
    FLOODNET_VQA_ID,
    FLOODNET_VQA_REVISION,
    LADI_QUESTIONS,
    LADI_V2_ID,
    LADI_V2_REVISION,
    image_data_url,
    question_kind,
)
from hsea.data.splits import split_for_seed

HF_DATASET_ID = "sanggusti/dmb-flywheel-v2"

FLOODNET_VQA_SPLIT = "train"  # #19 evals use validation; v2 trains on train.
LADI_V2_SPLIT = "train"  # #19 evals use test; v2 trains on train.

DISASTERM3_ID = "Kingdrone-Junjue/DisasterM3"
DISASTERM3_REVISION = "1da6da73fad6bd100454c099edd0cee972823810"
DISASTERM3_TRAIN_FILE = "DisasterM3_Instruct/train_release.json"
DISASTERM3_EXCLUDED_TASK = "Referring Expression Segmentation"

CRISITEXT_ID = "LanD-FBK/crisitext"
CRISITEXT_REVISION = "33c5545cd298527ae362cdd34a289486f6656dd6"
CRISITEXT_SPLIT = "train"
CRISITEXT_GUIDELINES_FILE = "fema_guidelines.json"
CRISITEXT_REAL_SOURCES = ("FEMA", "GTD")  # 'generated' rows are synthetic.

HUMAID_ID = "QCRI/HumAID-all"
HUMAID_REVISION = "93470f414471ae7464f4080b8a5a2b4f0f3226f5"
HUMAID_SPLIT = "train"

CRISISMMD_ID = "QCRI/CrisisMMD"
CRISISMMD_REVISION = "10a5626ba112ab9c50369c473a2062bcf205d708"
CRISISMMD_SPLIT = "train"
CRISISMMD_CONFIGS = ("humanitarian", "informative", "damage")
CRISISMMD_SEVERITY = {
    "severe_damage": "severe",
    "mild_damage": "mild",
    "little_or_no_damage": "none",
}

V2_SOURCES = (
    "floodnet-vqa",
    "ladi-v2",
    "disasterm3",
    "crisitext",
    "humaid",
    "crisismmd",
    "l2",
)


def _footage_sft(
    corpus: str,
    question: str,
    answer: str,
    image_ref: dict,
    extra: dict,
) -> dict:
    """One text SFT sample: question + ``[frame omitted]`` marker (v1 style).

    Pixels stay upstream; ``provenance`` carries the rehydration pointer.
    """
    return {
        "messages": [
            {"role": "user", "content": f"{question}\n[frame omitted]"},
            {"role": "assistant", "content": answer},
        ],
        "source": corpus,
        "provenance": {**image_ref, **extra},
    }


def floodnet_row_to_sample(row: dict, stream_index: int) -> dict:
    """Shape one ljx620 webdataset row (same layout as #19 footage evals)."""
    convo = row["json"]["conversations"]
    question = convo[0]["value"].replace("<image>\n", "")
    answer = convo[1]["value"]
    meta = row["json"]["meta"]
    _url, digest = image_data_url(bytes(row["0.img"]))
    return _footage_sft(
        "floodnet-vqa",
        question,
        answer,
        {
            "hf_dataset": FLOODNET_VQA_ID,
            "revision": FLOODNET_VQA_REVISION,
            "split": FLOODNET_VQA_SPLIT,
            "stream_index": stream_index,
            "image_sha256": digest,
        },
        {
            "image_id": meta.get("image_id", ""),
            "question_id": meta.get("question_id", stream_index),
            "question_type": meta.get("question_type", ""),
            "kind": question_kind(meta.get("question_type", "")),
        },
    )


def ladi_row_to_samples(row: dict, stream_index: int) -> list[dict]:
    """All :data:`LADI_QUESTIONS` per image (CC-BY-4.0, refs only)."""
    import io  # noqa: PLC0415 (kept with the heavy path)

    buf = io.BytesIO()
    row["image"].save(buf, format="PNG")
    _url, digest = image_data_url(buf.getvalue())
    samples = []
    for column, question, kind in LADI_QUESTIONS:
        if column not in row:
            raise ValueError(f"ladi-v2 row {stream_index} missing column {column!r}")
        samples.append(
            _footage_sft(
                "ladi-v2",
                question,
                "yes" if row[column] else "no",
                {
                    "hf_dataset": LADI_V2_ID,
                    "revision": LADI_V2_REVISION,
                    "split": LADI_V2_SPLIT,
                    "stream_index": stream_index,
                    "image_sha256": digest,
                },
                {"column": column, "kind": kind},
            )
        )
    return samples


def disasterm3_row_to_sample(row: dict, row_index: int) -> dict | None:
    """One DisasterM3 instruct row -> SFT sample, or None if excluded.

    Referring-seg rows carry mask paths instead of text answers, so they
    need vision SFT and are excluded here (counted in the manifest).
    """
    if row.get("task") == DISASTERM3_EXCLUDED_TASK:
        return None
    prompt = row.get("prompts")
    answer = row.get("training_answer") or row.get("ground_truth")
    if not prompt or not answer:
        raise ValueError(f"disasterm3 row {row_index} lacks prompt/answer text")
    user = prompt
    if row.get("options_str"):
        user += f"\nOptions: {row['options_str']}"
    user += "\n[frames omitted]"
    image_type = row.get("post_image_type") or row.get("image_type", "")
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "source": "disasterm3",
        "provenance": {
            "hf_dataset": DISASTERM3_ID,
            "revision": DISASTERM3_REVISION,
            "file": DISASTERM3_TRAIN_FILE,
            "row_index": row_index,
            "task": row.get("task", ""),
            "pre_image": row.get("pre_image_path", ""),
            "post_image": row.get("post_image_path", ""),
            "post_image_type": image_type,
        },
    }


def _parsed_list(raw: object, field: str, scenario: str) -> list:
    if isinstance(raw, list):
        return raw
    try:
        value = ast.literal_eval(raw)  # CSV stores repr-encoded lists.
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"crisitext {scenario} has unparsable {field}") from exc
    if not isinstance(value, list):
        raise ValueError(f"crisitext {scenario} {field} is not a list")
    return value


def crisitext_row_to_samples(row: dict, stream_index: int) -> list[dict]:
    """One SFT sample per warning message; synthetic rows excluded (None)."""
    scenario = row.get("scenario_id", f"row-{stream_index}")
    if row.get("source") not in CRISITEXT_REAL_SOURCES:
        return []
    guidelines = _parsed_list(row.get("guidelines"), "guidelines", scenario)
    messages = _parsed_list(row.get("messages"), "messages", scenario)
    bulletin = row.get("original_description", "")
    samples = []
    for msg in messages:
        samples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Source: {row.get('source')}\n"
                            f"Applicable FEMA guidelines: "
                            f"{', '.join(guidelines)}\n"
                            f"Situation bulletin:\n{bulletin}\n"
                            f"Event: {msg.get('event', '')}"
                        ),
                    },
                    {"role": "assistant", "content": msg.get("message", "")},
                ],
                "source": "crisitext",
                "provenance": {
                    "hf_dataset": CRISITEXT_ID,
                    "revision": CRISITEXT_REVISION,
                    "split": CRISITEXT_SPLIT,
                    "stream_index": stream_index,
                    "scenario_id": scenario,
                    "event_id": msg.get("event_id", ""),
                    "guidelines": list(guidelines),
                    "guidelines_file": (
                        f"{CRISITEXT_GUIDELINES_FILE}@{CRISITEXT_REVISION}"
                    ),
                },
            }
        )
    return samples


def humaid_row_to_record(row: dict, stream_index: int) -> dict:
    """Labels-only record. The tweet text is never read or stored."""
    category = row.get("class_label")
    if not category:
        raise ValueError(f"humaid row {stream_index} lacks class_label")
    return {
        "id": f"humaid-train-{stream_index:06d}",
        "labels": {"humanitarian": category},
    }


def join_crisismmd(
    humanitarian: dict[str, str],
    informative: dict[str, str],
    damage: dict[str, str],
    order: list[str],
) -> tuple[list[dict], dict[str, int]]:
    """Join the three CrisisMMD task configs by tweet_id (first-seen wins).

    A pair needs humanitarian + informative labels; severity is optional
    (the #20 builder falls back on unstated severity).
    """
    records = []
    dropped = 0
    for tweet_id in order:
        cat = humanitarian.get(tweet_id)
        info = informative.get(tweet_id)
        if cat is None or info is None:
            dropped += 1
            continue
        labels: dict = {
            "informative": info == "informative",
            "humanitarian": cat,
        }
        sev = damage.get(tweet_id)
        if sev is not None:
            labels["severity"] = CRISISMMD_SEVERITY[sev]
        records.append({"id": tweet_id, "labels": labels})
    return records, {"tweets_seen": len(order), "joined": len(records),
                     "dropped": dropped}


def stream_floodnet(cap: int) -> tuple[list[dict], dict]:
    """First ``cap`` train VQA pairs in stream order (pinned revision)."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, export-time only)

    ds = load_dataset(
        FLOODNET_VQA_ID, split=FLOODNET_VQA_SPLIT,
        revision=FLOODNET_VQA_REVISION, streaming=True,
    )
    samples = []
    for i, row in enumerate(ds):
        if i >= cap:
            break
        samples.append(floodnet_row_to_sample(row, i))
    return samples, {"streamed": len(samples), "kept": len(samples)}


def stream_ladi(max_images: int) -> tuple[list[dict], dict]:
    """12 QA pairs per image for the first ``max_images`` (pinned rev)."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, export-time only)

    ds = load_dataset(
        LADI_V2_ID, split=LADI_V2_SPLIT, revision=LADI_V2_REVISION, streaming=True
    )
    samples = []
    images = 0
    for i, row in enumerate(ds):
        if i >= max_images:
            break
        samples.extend(ladi_row_to_samples(row, i))
        images += 1
    return samples, {"images": images, "kept": len(samples)}


def fetch_disasterm3(cap: int) -> tuple[list[dict], dict]:
    """Task-balanced first-``cap`` non-referring instruct rows, file order.

    The release JSON is task-grouped, so the cap is split evenly across
    the eight text-answer tasks (deterministic per-task quotas).
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = hf_hub_download(
        DISASTERM3_ID, DISASTERM3_TRAIN_FILE, repo_type="dataset",
        revision=DISASTERM3_REVISION,
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = sorted({r.get("task", "") for r in rows} - {DISASTERM3_EXCLUDED_TASK})
    quota, remainder = divmod(cap, len(tasks))
    per_task = {t: quota + (1 if n < remainder else 0) for n, t in enumerate(tasks)}
    taken: dict[str, int] = {t: 0 for t in tasks}
    samples = []
    excluded = 0
    for i, row in enumerate(rows):
        if row.get("task") == DISASTERM3_EXCLUDED_TASK:
            excluded += 1
            continue
        task = row.get("task", "")
        if taken[task] >= per_task[task]:
            continue
        sample = disasterm3_row_to_sample(row, i)
        if sample is None:  # pragma: no cover (referring handled above)
            excluded += 1
            continue
        samples.append(sample)
        taken[task] += 1
    return samples, {
        "rows_total": len(rows), "kept": len(samples),
        "excluded_referring": excluded, "per_task": taken,
    }


def stream_crisitext(max_situations: int) -> tuple[list[dict], dict]:
    """First ``max_situations`` real-source situations, all their messages."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, export-time only)

    ds = load_dataset(
        CRISITEXT_ID, split=CRISITEXT_SPLIT, revision=CRISITEXT_REVISION,
        streaming=True,
    )
    samples = []
    situations = 0
    skipped_synthetic = 0
    for i, row in enumerate(ds):
        if situations >= max_situations:
            break
        if row.get("source") not in CRISITEXT_REAL_SOURCES:
            skipped_synthetic += 1
            continue
        samples.extend(crisitext_row_to_samples(row, i))
        situations += 1
    return samples, {
        "situations": situations, "kept": len(samples),
        "skipped_synthetic": skipped_synthetic,
    }


def stream_humaid(cap: int) -> tuple[list[dict], dict]:
    """First ``cap`` train rows as labels-only records (pinned revision)."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, export-time only)

    ds = load_dataset(
        HUMAID_ID, split=HUMAID_SPLIT, revision=HUMAID_REVISION, streaming=True
    )
    records = []
    for i, row in enumerate(ds):
        if i >= cap:
            break
        records.append(humaid_row_to_record(row, i))
    return records, {"streamed": len(records)}


def stream_crisismmd(cap: int) -> tuple[list[dict], dict]:
    """Join the three task configs by tweet_id; first ``cap`` joined."""
    from datasets import load_dataset  # noqa: PLC0415 (heavy, export-time only)

    labels: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for config in CRISISMMD_CONFIGS:
        ds = load_dataset(
            CRISISMMD_ID, name=config, split=CRISISMMD_SPLIT,
            revision=CRISISMMD_REVISION, streaming=True,
        )
        names = ds.features["label"].names
        seen: dict[str, str] = {}
        for row in ds:
            tweet = row["tweet_id"]
            if config == "humanitarian" and tweet not in seen:
                order.append(tweet)
            seen.setdefault(tweet, names[row["label"]])
        labels[config] = seen
    records, stats = join_crisismmd(
        labels["humanitarian"], labels["informative"], labels["damage"], order
    )
    stats["cap"] = cap
    return records[:cap], stats


def export_l2(
    log_glob: str, split: str, min_composite: float
) -> tuple[list[dict], list[dict], dict]:
    """SFT (success-filtered, non-mock) + RLVR (all scored) from eval logs.

    Samples outside ``split`` are skipped and counted (multi-log export
    must not abort on one off-split episode).
    """
    import glob  # noqa: PLC0415

    from inspect_ai.log import read_eval_log  # noqa: PLC0415

    sft: list[dict] = []
    rlvr: list[dict] = []
    stats = {"logs": 0, "episodes": 0, "sft_pairs": 0, "skipped_off_split": 0,
             "skipped_mock_sft": 0, "teachers": {}}
    for path in sorted(glob.glob(log_glob)):
        log = read_eval_log(path)
        teacher = str(getattr(getattr(log, "eval", None), "model", "unknown"))
        stats["logs"] += 1
        stats["teachers"][teacher] = stats["teachers"].get(teacher, 0) + 1
        for sample in log.samples or []:
            meta = dict(sample.metadata or {})
            seed = meta.get("seed")
            score = (sample.scores or {}).get("mission_score")
            if score is None or not isinstance(seed, int):
                continue
            try:
                if split_for_seed(seed) != split:
                    stats["skipped_off_split"] += 1
                    continue
            except ValueError:
                stats["skipped_off_split"] += 1
                continue
            composite = float(score.value)
            breakdown = dict(getattr(score, "metadata", None) or {})
            stats["episodes"] += 1
            rlvr.append(
                {
                    "sample_id": sample.id,
                    "scenario": meta.get("scenario"),
                    "seed": seed,
                    "world_hash": meta.get("world_hash"),
                    "rewards": breakdown,
                    "composite": composite,
                    "teacher": teacher,
                    "log": Path(path).name,
                    "l2": None,
                    "source": "l2",
                    "provenance": {"origin": "logs"},
                }
            )
            if composite < min_composite:
                continue
            pairs = messages_to_sft_pairs(list(sample.messages or []))
            if not pairs:
                continue
            if teacher == "mockllm/model":
                stats["skipped_mock_sft"] += len(pairs)
                continue
            for pair in pairs:
                sft.append(
                    {
                        "messages": pair,
                        "source": "l2",
                        "provenance": {
                            "origin": "logs",
                            "log": Path(path).name,
                            "teacher": teacher,
                            "sample_id": sample.id,
                            "scenario": meta.get("scenario"),
                            "seed": seed,
                            "composite": composite,
                        },
                    }
                )
    stats["sft_pairs"] = len(sft)
    return sft, rlvr, stats


def carry_v1(v1_dir: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Carry flywheel-v1 SFT/RLVR forward with v1-origin provenance."""
    base = Path(v1_dir)
    v1_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    teacher = v1_manifest.get("teacher", "unknown")
    sft = []
    for line in (base / "sft.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sft.append(
            {
                "messages": rec["messages"],
                "source": "l2",
                "provenance": {
                    "origin": "flywheel-v1",
                    "teacher": teacher,
                    "sample_id": rec.get("sample_id"),
                    "seed": rec.get("seed"),
                },
            }
        )
    rlvr = []
    for line in (base / "rlvr.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["source"] = "l2"
        rec["teacher"] = teacher
        rec["provenance"] = {"origin": "flywheel-v1"}
        rlvr.append(rec)
    return sft, rlvr, {
        "sft_pairs": len(sft), "episodes": len(rlvr),
        "teacher": teacher, "commit": v1_manifest.get("code_commit"),
    }


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


def _lib_versions() -> dict[str, str]:
    versions = {}
    for name in ("datasets", "huggingface_hub", "inspect_ai"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = "missing"
    return versions


def validate_sft(sample: dict) -> None:
    """Raise ValueError unless the sample matches the v2 SFT schema."""
    if not isinstance(sample.get("messages"), list) or not sample["messages"]:
        raise ValueError("sft sample needs a non-empty messages list")
    for turn in sample["messages"]:
        if turn.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"sft turn has bad role {turn.get('role')!r}")
        if "content" not in turn:
            raise ValueError("sft turn needs content")
    if sample.get("source") not in V2_SOURCES:
        raise ValueError(f"sft sample has bad source {sample.get('source')!r}")
    prov = sample.get("provenance")
    if not isinstance(prov, dict) or not prov:
        raise ValueError("sft sample needs provenance")


def validate_rlvr(record: dict) -> None:
    """Raise ValueError unless the record matches the v2 RLVR schema."""
    for key in ("sample_id", "scenario", "seed", "rewards", "composite", "source"):
        if key not in record:
            raise ValueError(f"rlvr record missing key {key!r}")
    if record["source"] != "l2":
        raise ValueError(f"rlvr record has bad source {record['source']!r}")
    if not isinstance(record["rewards"], dict):
        raise ValueError("rlvr record needs a rewards dict")


def write_export(
    sft: list[dict],
    dpo: list[dict],
    rlvr: list[dict],
    sources_meta: dict,
    out_dir: str | Path,
    split: str,
    export_command: str,
    dpo_counts: dict,
) -> dict:
    """Write ``sft/dpo/rlvr.jsonl`` + ``manifest.json``; returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows, validator in (
        ("sft.jsonl", sft, validate_sft),
        ("dpo.jsonl", dpo, validate_pair),
        ("rlvr.jsonl", rlvr, validate_rlvr),
    ):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                validator(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "issue": 7,
        "split": split,
        "code_commit": _code_commit(),
        "lib_versions": _lib_versions(),
        "export_command": export_command,
        "hf_dataset": HF_DATASET_ID,
        "sources": sources_meta,
        "totals": {"sft": len(sft), "dpo": len(dpo), "rlvr": len(rlvr)},
        "dpo_counts": dpo_counts,
        "files": {
            "sft.jsonl": _sha256(out / "sft.jsonl"),
            "dpo.jsonl": _sha256(out / "dpo.jsonl"),
            "rlvr.jsonl": _sha256(out / "rlvr.jsonl"),
        },
        "clearance": {
            "asserted": list(V2_SOURCES),
            "table": "hsea/data/clearance.py (#24 terms)",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def verify_export(out_dir: str | Path) -> dict:
    """Re-verify an export: clearance, hashes, schemas, counts. No network."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert_cleared_for_training(list(manifest["sources"]))
    seen: dict[str, int] = {}
    for name, validator, key in (
        ("sft.jsonl", validate_sft, "sft"),
        ("dpo.jsonl", validate_pair, "dpo"),
        ("rlvr.jsonl", validate_rlvr, "rlvr"),
    ):
        actual = _sha256(out / name)
        if manifest["files"][name] != actual:
            raise ValueError(f"{name} SHA-256 mismatch: {actual} != manifest")
        rows = [
            json.loads(line)
            for line in (out / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != manifest["totals"][key]:
            raise ValueError(f"{name} count {len(rows)} != manifest totals")
        for row in rows:
            validator(row)
        seen[key] = len(rows)
    per_source: dict[str, int] = {}
    for line in (out / "sft.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            source = json.loads(line)["source"]
            per_source[source] = per_source.get(source, 0) + 1
    return {"rows": seen, "sft_per_source": per_source,
            "split": manifest["split"], "hf_dataset": manifest["hf_dataset"]}


def _source_meta(source: str, stats: dict, **extra: object) -> dict:
    _status, terms, note = CLEARANCE[source]
    return {"license": terms, "clearance": f"cleared ({note})", **extra,
            "stats": stats}


def export_all(args: argparse.Namespace, export_command: str) -> dict:
    """Run the full curation (asserts clearance first, streams pinned revs)."""
    assert_cleared_for_training(list(V2_SOURCES))
    sft: list[dict] = []
    sources_meta: dict = {}

    floodnet, floodnet_stats = stream_floodnet(args.max_floodnet)
    sft.extend(floodnet)
    sources_meta["floodnet-vqa"] = _source_meta(
        "floodnet-vqa", floodnet_stats, hf_dataset=FLOODNET_VQA_ID,
        revision=FLOODNET_VQA_REVISION, split=FLOODNET_VQA_SPLIT,
        cap=args.max_floodnet,
    )

    ladi, ladi_stats = stream_ladi(args.max_ladi_images)
    sft.extend(ladi)
    sources_meta["ladi-v2"] = _source_meta(
        "ladi-v2", ladi_stats, hf_dataset=LADI_V2_ID,
        revision=LADI_V2_REVISION, split=LADI_V2_SPLIT,
        cap_images=args.max_ladi_images,
    )

    dm3, dm3_stats = fetch_disasterm3(args.max_disasterm3)
    sft.extend(dm3)
    sources_meta["disasterm3"] = _source_meta(
        "disasterm3", dm3_stats, hf_dataset=DISASTERM3_ID,
        revision=DISASTERM3_REVISION, file=DISASTERM3_TRAIN_FILE,
        cap=args.max_disasterm3,
    )

    crisitext, crisitext_stats = stream_crisitext(args.max_crisitext)
    sft.extend(crisitext)
    sources_meta["crisitext"] = _source_meta(
        "crisitext", crisitext_stats, hf_dataset=CRISITEXT_ID,
        revision=CRISITEXT_REVISION, split=CRISITEXT_SPLIT,
        cap_situations=args.max_crisitext,
    )

    humaid_records, humaid_stats = stream_humaid(args.max_humaid)
    crisismmd_records, crisismmd_stats = stream_crisismmd(args.max_crisismmd)
    dpo, dpo_counts = build_pairs(
        {"humaid": humaid_records, "crisismmd": crisismmd_records}, args.split
    )
    for pair in dpo:
        prov = pair["provenance"]
        if prov["source"] == "humaid":
            prov["hf_dataset"] = HUMAID_ID
            prov["revision"] = HUMAID_REVISION
        else:
            prov["hf_dataset"] = CRISISMMD_ID
            prov["revision"] = CRISISMMD_REVISION
    sources_meta["humaid"] = _source_meta(
        "humaid", humaid_stats, hf_dataset=HUMAID_ID,
        revision=HUMAID_REVISION, split=HUMAID_SPLIT, cap=args.max_humaid,
    )
    sources_meta["crisismmd"] = _source_meta(
        "crisismmd", crisismmd_stats, hf_dataset=CRISISMMD_ID,
        revision=CRISISMMD_REVISION, split=CRISISMMD_SPLIT,
        cap=args.max_crisismmd,
    )

    l2_sft_new, l2_rlvr_new, l2_stats = export_l2(
        args.l2_glob, args.split, args.min_composite
    )
    v1_sft, v1_rlvr, v1_stats = carry_v1(args.v1_dir)
    sft.extend(l2_sft_new)
    sft.extend(v1_sft)
    rlvr = [*l2_rlvr_new, *v1_rlvr]
    sources_meta["l2"] = _source_meta(
        "l2", {"fresh": l2_stats, "carry_v1": v1_stats},
        log_glob=args.l2_glob, v1_dir=str(args.v1_dir),
        min_composite=args.min_composite,
    )

    return write_export(sft, dpo, rlvr, sources_meta, args.out, args.split,
                        export_command, dpo_counts)


def plan(args: argparse.Namespace) -> dict:
    """Dry-run plan: clearance assert + pins/caps. No network, no writes."""
    assert_cleared_for_training(list(V2_SOURCES))
    return {
        "sources": list(V2_SOURCES),
        "pins": {
            "floodnet-vqa": f"{FLOODNET_VQA_ID}@{FLOODNET_VQA_REVISION[:8]}",
            "ladi-v2": f"{LADI_V2_ID}@{LADI_V2_REVISION[:8]}",
            "disasterm3": f"{DISASTERM3_ID}@{DISASTERM3_REVISION[:8]}",
            "crisitext": f"{CRISITEXT_ID}@{CRISITEXT_REVISION[:8]}",
            "humaid": f"{HUMAID_ID}@{HUMAID_REVISION[:8]}",
            "crisismmd": f"{CRISISMMD_ID}@{CRISISMMD_REVISION[:8]}",
            "l2": f"{args.l2_glob} + {args.v1_dir}",
        },
        "caps": {
            "floodnet": args.max_floodnet,
            "ladi_images": args.max_ladi_images,
            "disasterm3": args.max_disasterm3,
            "crisitext_situations": args.max_crisitext,
            "humaid": args.max_humaid,
            "crisismmd": args.max_crisismmd,
        },
        "hf_dataset": HF_DATASET_ID,
        "out": str(args.out),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curate flywheel-v2 datasets (#7).")
    parser.add_argument("--out", required=True, help="output dir")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--max-floodnet", type=int, default=1000)
    parser.add_argument("--max-ladi-images", type=int, default=200)
    parser.add_argument("--max-disasterm3", type=int, default=3000)
    parser.add_argument("--max-crisitext", type=int, default=300)
    parser.add_argument("--max-humaid", type=int, default=4000)
    parser.add_argument("--max-crisismmd", type=int, default=3000)
    parser.add_argument("--min-composite", type=float, default=0.5)
    parser.add_argument("--l2-glob", default="logs/*.eval")
    parser.add_argument("--v1-dir", default="data/flywheel-v1")
    parser.add_argument("--dry-run", action="store_true",
                        help="assert clearance + print plan, no writes")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an existing export")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            print(json.dumps(verify_export(args.out), indent=2))
            return
        if args.dry_run:
            print(json.dumps(plan(args), indent=2))
            return
        command = " ".join(["python -m hsea.data.curate_v2", *sys.argv[1:]])
        print(json.dumps(export_all(args, command), indent=2))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

