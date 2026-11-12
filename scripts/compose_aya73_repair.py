"""Compose the issue-73 Stage 4 SafeTune repair mixes.

Targets the two Phase-1 symptoms directly:

- red-team ASR regressed 0.633 (base) -> 0.833 (DPO): the repair mix
  adds the 80-row WJ-methodology disaster set (40 misuse/refuse + 40
  lookalike/comply) in the #73 DECISION safety-demo format.
- abstention retreated SFT 0.650 -> DPO 0.362 via verdict drift
  (repeated ACTION blocks, last-wins scoring): abstention cal demos
  re-teach the single-verdict ACTION contract, and KTO pairs each cal
  gold (desirable) with a wrong-label violation plus the REAL DPO
  drift completion on that prompt (undesirable, mined after G0).

#73 keeps its gate discipline: HarmBench + AIR-Bench rows NEVER train
(the 26 safety demos come from the same non-gate train half as
Phase 1), abstention eval-split rows NEVER train (cal only), and the
abstention ACTION footer is KEPT (no DECISION migration: the scorer
parses ACTION last-wins, and #74 showed migration does not take).

- Dispatch anchors: 200 footer-free flywheel chosen rows (stratified
  P1/P2/P3, raw situation user text = Phase-1 shape) hold the
  no-footer dispatch format.
- Dispatch debias: UltraFeedback-template debias strips the
  ``PRIORITY N`` headline prefix from BOTH sides of 500 repair DPO
  pairs, forcing the pair loss onto action content.

Usage:
    python scripts/compose_aya73_repair.py --out data/aya73
    python scripts/compose_aya73_repair.py finalize-kto --out data/aya73 \\
        --completions runs/aya73-followup/eval_dpo.jsonl
Output: ``repair_sft.jsonl``, ``repair_dpo.jsonl``, ``kto.jsonl``,
``repair_manifest.json``. Held-out sets (redteam, dispatch-pairs,
xstest, gate benches, abstention eval split) stay eval-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compose_aya73 import (  # noqa: E402
    ABSTENTION_SYSTEM,
    DISPATCH_SYSTEM,
    abstention_gold,
    abstention_rejected_footer,
    safety_demo_response,
    safety_rejected_response,
    split_safety,
)
from compose_aya74_repair import debias, stratified  # noqa: E402
from hsea.safety.data import SAFETY_SYSTEM  # noqa: E402

SEED = 73
N_DISPATCH_SFT = 200
N_DISPATCH_DPO = 500

KTO_NEGATIVE = {"escalate": "comply", "abstain": "comply",
                "comply": "abstain"}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _abstention_rows() -> list[dict]:
    return (load_jsonl(ROOT / "data" / "abstention" / "scenarios.jsonl")
            + load_jsonl(ROOT / "data" / "abstention" / "benign.jsonl"))


def compose_repair_sft() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    train_items, _ = split_safety()
    for bench, category, label, prompt in train_items:
        rows.append({
            "messages": [
                {"role": "system", "content": SAFETY_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": safety_demo_response(label, prompt)},
            ],
            "source": "hsea-safety",
            "provenance": {"mix": "aya73-repair-sft", "bench": bench,
                           "category": category, "label": label,
                           "footer": "DECISION", "gate": False},
        })
    n_safety = len(train_items)

    n_abst = 0
    for rec in _abstention_rows():
        if rec["split"] != "cal":
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": ABSTENTION_SYSTEM},
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": abstention_gold(rec)},
            ],
            "source": "hsea-abstention",
            "provenance": {"mix": "aya73-repair-sft", "id": rec["id"],
                           "label": rec["label"], "split": "cal",
                           "footer": "ACTION-single-verdict"},
        })
        n_abst += 1

    fly = load_jsonl(ROOT / "data" / "flywheel-v2" / "dpo.jsonl")
    anchors = stratified(fly, N_DISPATCH_SFT, seed=SEED)
    for rec in anchors:
        assert "DECISION:" not in rec["chosen"], "anchor has footer"
        assert "ACTION:" not in rec["chosen"], "anchor has footer"
        rows.append({
            "messages": [
                {"role": "system", "content": DISPATCH_SYSTEM},
                {"role": "user", "content": rec["situation"]},
                {"role": "assistant", "content": rec["chosen"]},
            ],
            "source": "flywheel-dispatch-anchor",
            "provenance": {"mix": "aya73-repair-sft",
                           "record_id": rec["provenance"]["record_id"],
                           "priority": rec["priority"],
                           "footer": "none-dispatch-format"},
        })

    n_wj = 0
    for rec in load_jsonl(ROOT / "data" / "wildjailbreak-method"
                          / "disaster_mix.jsonl"):
        demo = safety_demo_response(rec["label"], rec["prompt"])
        rows.append({
            "messages": [
                {"role": "system", "content": SAFETY_SYSTEM},
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": demo},
            ],
            "source": "wj-method",
            "provenance": {"mix": "aya73-repair-sft", "id": rec["id"],
                           "kind": rec["kind"], "label": rec["label"],
                           "footer": "DECISION"},
        })
        n_wj += 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    stats = {"safety": n_safety, "abstention_cal": n_abst,
             "dispatch_anchors": len(anchors), "wj_method": n_wj,
             "total": len(rows)}
    return rows, stats


def compose_repair_dpo() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    train_items, _ = split_safety()
    for bench, category, label, prompt in train_items:
        rows.append({
            "system": SAFETY_SYSTEM,
            "prompt": prompt,
            "chosen": safety_demo_response(label, prompt),
            "rejected": safety_rejected_response(label, prompt),
            "source": "hsea-safety",
            "provenance": {"mix": "aya73-repair-dpo", "bench": bench,
                           "category": category, "label": label,
                           "footer": "DECISION", "gate": False},
        })
    n_safety = len(train_items)

    fly = load_jsonl(ROOT / "data" / "flywheel-v2" / "dpo.jsonl")
    pairs = stratified(fly, N_DISPATCH_DPO, seed=SEED)
    for rec in pairs:
        chosen, rejected = debias(rec["chosen"]), debias(rec["rejected"])
        assert chosen and rejected, "debias emptied a side"
        rows.append({
            "system": DISPATCH_SYSTEM,
            "prompt": rec["situation"],
            "chosen": chosen,
            "rejected": rejected,
            "source": "flywheel-dispatch-debiased",
            "provenance": {"mix": "aya73-repair-dpo",
                           "record_id": rec["provenance"]["record_id"],
                           "priority": rec["priority"],
                           "debias": "priority-prefix-strip"},
        })
    stats = {"safety": n_safety, "dispatch_debiased": len(pairs),
             "total": len(rows)}
    return rows, stats


def compose_kto() -> tuple[list[dict], dict]:
    """Binary abstention labels: cal gold (True) vs violation (False).

    Drift negatives (real DPO multi-ACTION completions) are appended
    by ``finalize_kto`` once G0 generations exist.
    """
    rows: list[dict] = []
    for rec in _abstention_rows():
        if rec["split"] != "cal":
            continue
        gold = rec["label"]
        rows.append({
            "prompt": rec["prompt"],
            "completion": abstention_gold(rec),
            "label": True,
            "provenance": {"mix": "aya73-kto", "id": rec["id"],
                           "label": gold, "side": "desirable"},
        })
        wrong = KTO_NEGATIVE[gold]
        rows.append({
            "prompt": rec["prompt"],
            "completion": rec["rationale"] + "\n"
            + abstention_rejected_footer(gold),
            "label": False,
            "provenance": {"mix": "aya73-kto", "id": rec["id"],
                           "label": gold, "side": "undesirable",
                           "violation": wrong},
        })
    stats = {"prompts": len(rows) // 2, "total": len(rows),
             "desirable": sum(1 for r in rows if r["label"]),
             "undesirable": sum(1 for r in rows if not r["label"]),
             "drift_negatives": 0}
    return rows, stats


def finalize_kto(completions_path: Path) -> tuple[list[dict], int]:
    """Append real DPO drift completions as KTO undesirables.

    A cal prompt contributes its DPO completion when the completion
    shows verdict drift (>= 2 ACTION footers) AND the last verdict
    misses gold. Returns (drift_rows, n_cal_with_completions).
    """
    import re

    from hsea.scorers.abstention import parse_verdict

    action_re = re.compile(r"^\s*ACTION\s*:\s*(comply|escalate|abstain)\b",
                           re.IGNORECASE | re.MULTILINE)
    comps = {}
    for line in open(completions_path):
        rec = json.loads(line)
        comps[rec["key"]] = rec.get("completion", "")
    cal = {r["id"]: r for r in _abstention_rows() if r["split"] == "cal"}
    drift_rows: list[dict] = []
    n_seen = 0
    for cid, rec in sorted(cal.items()):
        comp = comps.get(f"abstention:{cid}")
        if comp is None:
            continue
        n_seen += 1
        actions = action_re.findall(comp)
        pred, _ = parse_verdict(comp)
        if len(actions) >= 2 and pred != rec["label"]:
            drift_rows.append({
                "prompt": rec["prompt"],
                "completion": comp,
                "label": False,
                "provenance": {"mix": "aya73-kto", "id": cid,
                               "label": rec["label"], "side": "undesirable",
                               "violation": "dpo-verdict-drift",
                               "n_actions": len(actions),
                               "last_verdict": pred},
            })
    return drift_rows, n_seen


def cmd_compose(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sft_rows, sft_stats = compose_repair_sft()
    dpo_rows, dpo_stats = compose_repair_dpo()
    kto_rows, kto_stats = compose_kto()

    for name, rows in (("repair_sft.jsonl", sft_rows),
                      ("repair_dpo.jsonl", dpo_rows),
                      ("kto.jsonl", kto_rows)):
        with open(out / name, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    manifest = {
        "issue": 73,
        "stage": 4,
        "seed": SEED,
        "gate_discipline": "HarmBench+AIR-Bench NEVER train (26 demos "
                           "from the Phase-1 non-gate train half); "
                           "abstention eval split NEVER trains (cal "
                           "only); ACTION footer kept (scorer parses "
                           "ACTION last-wins)",
        "targets": ["redteam ASR 0.633->0.833 regression (WJ-method "
                    "disaster mix)", "abstention SFT 0.650 -> DPO 0.362 "
                    "verdict drift (single-verdict demos + KTO drift "
                    "negatives)"],
        "dispatch_debias": "priority-prefix-strip on both pair sides",
        "sft": sft_stats,
        "dpo": dpo_stats,
        "kto": kto_stats,
        "clearance": {
            "wj_upstream": "allenai/wildjailbreak is access-gated "
                           "(personal click-through; out of scope per "
                           "license-clearance-24 §3)",
            "wj_method": "curator-written methodology subset, "
                         "data/wildjailbreak-method/, 80 rows",
        },
        "held_out": ["data/redteam/attacks.jsonl",
                     "data/dispatch-pairs/dpo.jsonl",
                     "data/xstest/lookalikes.jsonl",
                     "harmbench+airbench gate rows",
                     "abstention eval split"],
    }
    with open(out / "repair_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"repair_sft={sft_stats} repair_dpo={dpo_stats} "
          f"kto={kto_stats} -> {out}")


def cmd_finalize_kto(out: Path, completions: Path) -> None:
    drift_rows, n_seen = finalize_kto(completions)
    kto_path = out / "kto.jsonl"
    rows = load_jsonl(kto_path)
    have = {r["provenance"].get("id") for r in rows
            if r["provenance"].get("violation") == "dpo-verdict-drift"}
    fresh = [r for r in drift_rows
             if r["provenance"]["id"] not in have]
    with open(kto_path, "a") as fh:
        for row in fresh:
            fh.write(json.dumps(row) + "\n")
    manifest_path = out / "repair_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["kto"]["drift_negatives"] = (
        manifest["kto"].get("drift_negatives", 0) + len(fresh))
    manifest["kto"]["total"] = len(rows) + len(fresh)
    manifest["kto"]["undesirable"] = sum(
        1 for r in rows + fresh if not r["label"])
    manifest["kto"]["drift_source"] = str(completions)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"kto finalize: cal_seen={n_seen} drift_found={len(drift_rows)} "
          f"appended={len(fresh)} total={len(rows) + len(fresh)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="compose repair mixes")
    parser.add_argument("--out", default="data/aya73")
    parser.add_argument("command", nargs="?", default="compose",
                        choices=("compose", "finalize-kto"))
    parser.add_argument("--completions", default="",
                        help="eval_dpo.jsonl for finalize-kto")
    args = parser.parse_args(argv)
    if args.command == "finalize-kto":
        if not args.completions:
            parser.error("finalize-kto needs --completions")
        cmd_finalize_kto(ROOT / args.out, Path(args.completions))
    else:
        cmd_compose(ROOT / args.out)


if __name__ == "__main__":
    main()
