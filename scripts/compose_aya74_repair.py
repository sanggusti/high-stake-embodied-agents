"""Compose the issue-74 Stage 4 SafeTune repair mixes.

Repairs the reported T5 dispatch regression (footer leakage + hedged
actions) and the drift-audit Recover verdict with a small constrained
re-tune mix:

- Footer unification: ONE footer contract everywhere, ``DECISION:``
  with the 4-way ``refuse|escalate|abstain|comply`` vocab. Abstention
  demos migrate from ``ACTION:`` to ``DECISION:`` (labels already
  align); off-label values (``standdown``, ``notify_coordinator``,
  ``prioritize``, ``draft``...) disappear by construction.
- Dispatch anchors: footer-free dispatch SFT demos (flywheel DPO
  chosen rows) re-anchor the no-footer dispatch format.
- Dispatch debias: UltraFeedback-template debias strips the
  ``PRIORITY N`` headline prefix from BOTH sides of repair DPO
  pairs, forcing the pair loss onto action content.
- WildJailbreak-method disaster mix: 40 misuse (refuse) + 40
  lookalike (comply) curator-written rows (upstream is access-gated;
  see ``data/wildjailbreak-method/README.md``).
- KTO split: binary abstention labels (desirable demo vs wrong-label
  violation) for ``KTOTrainer``.

Usage:
    python scripts/compose_aya74_repair.py --out data/aya74
Output: ``repair_sft.jsonl``, ``repair_dpo.jsonl``, ``kto.jsonl``,
``repair_manifest.json``. Held-out sets (redteam, dispatch-pairs,
xstest) stay eval-only; dispatch anchors come from flywheel TRAIN
pairs, never the held-out dispatch-pairs file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compose_aya74 import (  # noqa: E402
    ABSTAIN_CONFIDENCE,
    safety_demo,
    safety_violation,
)

SEED = 74
N_DISPATCH_SFT = 200
N_DISPATCH_DPO = 500

PRIORITY_PREFIX = re.compile(r"^PRIORITY [123] \u2014 [^.]+\.\s*")

# Wrong-label violation map for KTO negatives: acting without approval
# on escalate/abstain rows, abstaining on comply rows.
KTO_NEGATIVE = {"escalate": "comply", "abstain": "comply",
                "comply": "abstain"}


def unified_demo(label: str, rationale: str) -> str:
    """Abstention demo under the UNIFIED footer contract (DECISION)."""
    if label == "escalate":
        tail = ("I propose a provisional course and require on-scene "
                "authority to approve it before anyone acts.")
    elif label == "abstain":
        tail = ("No safe proposal exists, so I decline the task and "
                "redirect to qualified responders.")
    else:
        tail = "This is safe to handle directly, so I act now."
    return (f"{rationale} {tail}\nDECISION: {label}\n"
            f"CONFIDENCE: {ABSTAIN_CONFIDENCE[label]}")


def debias(text: str) -> str:
    """Strip the PRIORITY-headline template prefix (both pair sides)."""
    return PRIORITY_PREFIX.sub("", text).strip()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratified(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Proportional sample by ``priority`` (P1/P2/P3), seeded."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for rec in rows:
        buckets.setdefault(rec.get("priority", "?"), []).append(rec)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    total = len(rows)
    quotas = {k: max(1, round(n * len(v) / total))
              for k, v in buckets.items()}
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=quotas.get)  # type: ignore[arg-type]
        quotas[biggest] -= 1
    out = []
    for key in sorted(buckets):
        out.extend(buckets[key][:quotas[key]])
    rng.shuffle(out)
    assert len(out) == n, f"stratified {len(out)} != {n}"
    return out


def dispatch_prompt(situation: str) -> str:
    return (f"Situation: {situation}\n"
            f"Decide the dispatch priority and action.")


def compose_repair_sft() -> tuple[list[dict], dict]:
    from hsea.safety.data import ITEMS

    rows: list[dict] = []
    for bench, category, label, prompt in ITEMS:
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": safety_demo(label, prompt, category)},
            ],
            "source": "hsea-safety",
            "provenance": {"mix": "aya74-repair-sft", "bench": bench,
                           "category": category, "label": label,
                           "footer": "DECISION-unified"},
        })
    n_safety = len(ITEMS)

    n_abst = 0
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for rec in load_jsonl(ROOT / "data" / "abstention" / name):
            rows.append({
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant",
                     "content": unified_demo(rec["label"],
                                             rec.get("rationale", ""))},
                ],
                "source": "hsea-abstention",
                "provenance": {"mix": "aya74-repair-sft", "id": rec["id"],
                               "label": rec["label"],
                               "footer": "DECISION-unified"},
            })
            n_abst += 1

    fly = load_jsonl(ROOT / "data" / "flywheel-v2" / "dpo.jsonl")
    anchors = stratified(fly, N_DISPATCH_SFT)
    for rec in anchors:
        assert "DECISION:" not in rec["chosen"], "anchor has footer"
        assert "ACTION:" not in rec["chosen"], "anchor has footer"
        rows.append({
            "messages": [
                {"role": "user",
                 "content": dispatch_prompt(rec["situation"])},
                {"role": "assistant", "content": rec["chosen"]},
            ],
            "source": "flywheel-dispatch-anchor",
            "provenance": {"mix": "aya74-repair-sft",
                           "record_id": rec["provenance"]["record_id"],
                           "priority": rec["priority"],
                           "footer": "none-dispatch-format"},
        })

    n_wj = 0
    for rec in load_jsonl(ROOT / "data" / "wildjailbreak-method"
                          / "disaster_mix.jsonl"):
        demo = safety_demo(rec["label"], rec["prompt"], rec["category"])
        rows.append({
            "messages": [
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": demo},
            ],
            "source": "wj-method",
            "provenance": {"mix": "aya74-repair-sft", "id": rec["id"],
                           "kind": rec["kind"], "label": rec["label"],
                           "footer": "DECISION-unified"},
        })
        n_wj += 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    stats = {"safety": n_safety, "abstention": n_abst,
             "dispatch_anchors": len(anchors), "wj_method": n_wj,
             "total": len(rows)}
    return rows, stats


def compose_repair_dpo() -> tuple[list[dict], dict]:
    from hsea.safety.data import ITEMS

    rows: list[dict] = []
    for bench, category, label, prompt in ITEMS:
        rows.append({
            "prompt": prompt,
            "chosen": safety_demo(label, prompt, category),
            "rejected": safety_violation(label, prompt, category),
            "source": "hsea-safety",
            "provenance": {"mix": "aya74-repair-dpo", "bench": bench,
                           "category": category, "label": label,
                           "footer": "DECISION-unified"},
        })
    n_safety = len(ITEMS)

    fly = load_jsonl(ROOT / "data" / "flywheel-v2" / "dpo.jsonl")
    pairs = stratified(fly, N_DISPATCH_DPO)
    for rec in pairs:
        chosen, rejected = debias(rec["chosen"]), debias(rec["rejected"])
        assert chosen and rejected, "debias emptied a side"
        rows.append({
            "prompt": dispatch_prompt(rec["situation"]),
            "chosen": chosen,
            "rejected": rejected,
            "source": "flywheel-dispatch-debiased",
            "provenance": {"mix": "aya74-repair-dpo",
                           "record_id": rec["provenance"]["record_id"],
                           "priority": rec["priority"],
                           "debias": "priority-prefix-strip"},
        })
    stats = {"safety": n_safety, "dispatch_debiased": len(pairs),
             "total": len(rows)}
    return rows, stats


def compose_kto() -> tuple[list[dict], dict]:
    """Binary abstention labels: gold demo (True) vs violation (False)."""
    rows: list[dict] = []
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for rec in load_jsonl(ROOT / "data" / "abstention" / name):
            gold = rec["label"]
            rows.append({
                "prompt": rec["prompt"],
                "completion": unified_demo(gold, rec.get("rationale", "")),
                "label": True,
                "provenance": {"mix": "aya74-kto", "id": rec["id"],
                               "label": gold, "side": "desirable"},
            })
            wrong = KTO_NEGATIVE[gold]
            rows.append({
                "prompt": rec["prompt"],
                "completion": unified_demo(wrong, rec.get("rationale", "")),
                "label": False,
                "provenance": {"mix": "aya74-kto", "id": rec["id"],
                               "label": gold, "side": "undesirable",
                               "violation": wrong},
            })
    stats = {"prompts": len(rows) // 2, "total": len(rows),
             "desirable": sum(1 for r in rows if r["label"]),
             "undesirable": sum(1 for r in rows if not r["label"])}
    return rows, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="compose repair mixes")
    parser.add_argument("--out", default="data/aya74")
    args = parser.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    sft_rows, sft_stats = compose_repair_sft()
    dpo_rows, dpo_stats = compose_repair_dpo()
    kto_rows, kto_stats = compose_kto()

    with open(out / "repair_sft.jsonl", "w") as fh:
        for row in sft_rows:
            fh.write(json.dumps(row) + "\n")
    with open(out / "repair_dpo.jsonl", "w") as fh:
        for row in dpo_rows:
            fh.write(json.dumps(row) + "\n")
    with open(out / "kto.jsonl", "w") as fh:
        for row in kto_rows:
            fh.write(json.dumps(row) + "\n")
    manifest = {
        "issue": 74,
        "stage": 4,
        "seed": SEED,
        "footer_contract": "DECISION-unified (refuse|escalate|abstain|"
                           "comply); abstention migrated off ACTION",
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
                     "data/xstest/lookalikes.jsonl"],
    }
    with open(out / "repair_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"repair_sft={sft_stats} repair_dpo={dpo_stats} "
          f"kto={kto_stats} -> {out}")


if __name__ == "__main__":
    main()
