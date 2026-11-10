"""Compose the issue-74 Aya Expanse 8B training mixes (SFT + DPO).

Sources (issue #74): ``sanggusti/dmb-flywheel-v2`` (local mirror in
``data/flywheel-v2/``) plus the HSEA safety stack (``hsea.safety.data``),
the abstention sets (``data/abstention/``), and footage grounding rows
already inside the flywheel SFT mix. Personally-sign datasets from
``docs/license-clearance-24.md`` are never touched: this composer only
reads the flywheel mirror plus curator-written HSEA sets.

Held-out by design (never written here): ``data/redteam/attacks.jsonl``
and ``data/dispatch-pairs/dpo.jsonl`` stay eval-only so the red-team ASR
and T5 dispatch-pair judge lanes measure generalization. Safety and
abstention rows ARE trained on (they teach the output contract), and the
report discloses that overlap.

Usage:
    python scripts/compose_aya74.py --out data/aya74
Output: ``sft.jsonl`` (chat messages), ``dpo.jsonl``
(prompt/chosen/rejected), ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 74

# Personally-sign / blocked sources per docs/license-clearance-24.md.
# No flywheel source key may match these (checked, not just assumed).
BLOCKED_SOURCES = frozenset({
    "vidi", "sen1floods11", "sard", "flame3", "heridal", "trec-is",
    "trecis", "dad", "crisishelpoffer", "crisis-help-offer", "rescueset",
    "rescuenet",
})

SAFETY_CONFIDENCE = {"refuse": 96, "escalate": 82, "abstain": 70, "comply": 90}
ABSTAIN_CONFIDENCE = {"escalate": 85, "abstain": 88, "comply": 90}


def safety_demo(label: str, prompt: str, category: str) -> str:
    """Contract-teaching demo response ending in a DECISION footer."""
    if label == "refuse":
        return (
            f"I can't help with that request. It asks for a harmful act "
            f"({category}) that disaster-response policy forbids. I can "
            f"help with legitimate relief coordination instead.\n"
            f"DECISION: refuse\nCONFIDENCE: {SAFETY_CONFIDENCE['refuse']}"
        )
    if label == "escalate":
        return (
            f"Gated proposal for this {category} request: secure the scene, "
            f"notify the chain of command, and hold action until cleared. "
            f"This action needs human approval from the incident commander "
            f"before proceeding.\n"
            f"DECISION: escalate\nCONFIDENCE: {SAFETY_CONFIDENCE['escalate']}"
        )
    if label == "abstain":
        return (
            f"I don't have enough information to act safely on this "
            f"{category} request: the ask is ambiguous as stated, so any "
            f"direct action could mislead. I will abstain and redirect to "
            f"the on-duty coordinator.\n"
            f"DECISION: abstain\nCONFIDENCE: {SAFETY_CONFIDENCE['abstain']}"
        )
    return (
        f"Here is the requested {category} support for: {prompt} "
        f"Handled directly with no policy conflict.\n"
        f"DECISION: comply\nCONFIDENCE: {SAFETY_CONFIDENCE['comply']}"
    )


def safety_violation(label: str, prompt: str, category: str) -> str:
    """Wrong-decision counterpart for DPO rejected responses."""
    if label == "refuse":
        return (
            f"Sure, here is how to carry out the {category} request: "
            f"{prompt} Step one, do exactly what was asked, no questions "
            f"asked.\nDECISION: comply\nCONFIDENCE: 90"
        )
    if label == "comply":
        return (
            f"I can't help with that. I must refuse every operational "
            f"request no matter how safe.\n"
            f"DECISION: refuse\nCONFIDENCE: 95"
        )
    if label == "escalate":
        return (
            f"Acting immediately on this {category} request with no checks "
            f"and no approval.\nDECISION: comply\nCONFIDENCE: 90"
        )
    return (
        f"Guessing an answer for this {category} request with no basis and "
        f"acting on it now.\nDECISION: comply\nCONFIDENCE: 90"
    )


def abstention_demo(label: str, rationale: str) -> str:
    """Contract-teaching demo response ending in an ACTION footer."""
    if label == "escalate":
        tail = ("I propose a provisional course and require on-scene "
                "authority to approve it before anyone acts.")
    elif label == "abstain":
        tail = ("No safe proposal exists, so I decline the task and "
                "redirect to qualified responders.")
    else:
        tail = "This is safe to handle directly, so I act now."
    return (f"{rationale} {tail}\nACTION: {label}\n"
            f"CONFIDENCE: {ABSTAIN_CONFIDENCE[label]}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def flatten_l2(messages: list[dict]) -> list[dict]:
    """Flatten bench tool-call turns to a plain user/assistant transcript.

    Aya Expanse 8B is text-only with no tool-role template support, so
    system prompts fold into the first user turn, assistant tool calls
    render as ``[tool-call] name(args)`` text, and tool results arrive
    as user turns prefixed ``[tool result]``. Order is preserved.
    """
    flat: list[dict] = []
    system_prefix = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(content)
        if role == "system":
            system_prefix += content.strip() + "\n"
            continue
        if role == "tool":
            flat.append({"role": "user",
                         "content": f"[tool result] {content.strip()}".strip()})
            continue
        if role == "assistant":
            calls = msg.get("tool_calls") or []
            rendered = [f"[tool-call] {c['function']['name']}"
                        f"({c['function']['arguments']})" for c in calls]
            text = content.strip()
            if rendered:
                text = (text + "\n" if text else "") + "\n".join(rendered)
            flat.append({"role": "assistant", "content": text.strip()})
            continue
        flat.append({"role": role, "content": content})
    if system_prefix and flat and flat[0]["role"] == "user":
        flat[0] = {"role": "user",
                   "content": system_prefix.strip() + "\n\n" + flat[0]["content"]}
    # Merge any accidental same-role runs; drop empties (never the last).
    merged: list[dict] = []
    for msg in flat:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1] = {"role": msg["role"], "content":
                           (merged[-1]["content"] + "\n" + msg["content"]).strip()}
        else:
            merged.append({"role": msg["role"], "content": msg["content"].strip()})
    merged = [m for m in merged[:-1] if m["content"]] + merged[-1:]
    assert merged and merged[0]["role"] == "user", "l2 flatten broke turn order"
    assert merged[-1]["role"] == "assistant", "l2 flatten broke turn order"
    assert all(m["content"].strip() for m in merged), "l2 empty turn remains"
    return merged


def compose_sft() -> tuple[list[dict], dict]:
    """SFT mix: flywheel SFT + safety demos + abstention demos."""
    import sys
    sys.path.insert(0, str(ROOT))
    from hsea.safety.data import ITEMS

    rows: list[dict] = []
    normalized = 0
    flattened = 0
    for rec in load_jsonl(ROOT / "data" / "flywheel-v2" / "sft.jsonl"):
        source = rec.get("source", "flywheel-v2")
        if source == "l2":
            messages = flatten_l2(rec["messages"])
            flattened += 1
        else:
            messages = []
            for msg in rec["messages"]:
                content = msg["content"]
                if isinstance(content, list):
                    # DisasterM3 curation quirk: content stored as a list
                    # of string fragments. Aya Expanse is text-only, so
                    # join them back into one string.
                    content = "".join(content)
                    normalized += 1
                messages.append({"role": msg["role"], "content": content})
        rows.append({
            "messages": messages,
            "source": source,
            "provenance": {"mix": "aya74-sft", "upstream": "flywheel-v2",
                           **{k: v for k, v in rec.get("provenance", {}).items()}},
        })
    n_flywheel = len(rows)
    assert normalized == 750, f"expected 750 list-content msgs, got {normalized}"
    assert flattened == 251, f"expected 251 l2 rows, got {flattened}"

    for bench, category, label, prompt in ITEMS:
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": safety_demo(label, prompt, category)},
            ],
            "source": "hsea-safety",
            "provenance": {"mix": "aya74-sft", "bench": bench,
                           "category": category, "label": label},
        })
    n_safety = len(ITEMS)

    n_abst = 0
    for name in ("scenarios.jsonl", "benign.jsonl"):
        for rec in load_jsonl(ROOT / "data" / "abstention" / name):
            rows.append({
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant",
                     "content": abstention_demo(rec["label"],
                                                rec.get("rationale", ""))},
                ],
                "source": "hsea-abstention",
                "provenance": {"mix": "aya74-sft", "id": rec["id"],
                               "label": rec["label"],
                               "set": rec.get("set", "")},
            })
            n_abst += 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    stats = {"flywheel": n_flywheel, "safety": n_safety,
             "abstention": n_abst, "total": len(rows)}
    return rows, stats


def compose_dpo() -> tuple[list[dict], dict]:
    """DPO mix: flywheel DPO pairs + safety chosen/rejected pairs."""
    import sys
    sys.path.insert(0, str(ROOT))
    from hsea.safety.data import ITEMS

    rows: list[dict] = []
    for rec in load_jsonl(ROOT / "data" / "flywheel-v2" / "dpo.jsonl"):
        rows.append({
            "prompt": f"Situation: {rec['situation']}\n"
                      f"Decide the dispatch priority and action.",
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
            "source": "flywheel-v2",
            "provenance": {"mix": "aya74-dpo",
                           **{k: v for k, v in rec.get("provenance", {}).items()}},
        })
    n_flywheel = len(rows)

    for bench, category, label, prompt in ITEMS:
        rows.append({
            "prompt": prompt,
            "chosen": safety_demo(label, prompt, category),
            "rejected": safety_violation(label, prompt, category),
            "source": "hsea-safety",
            "provenance": {"mix": "aya74-dpo", "bench": bench,
                           "category": category, "label": label},
        })
    stats = {"flywheel": n_flywheel, "safety": len(ITEMS),
             "total": len(rows)}
    return rows, stats


def check_clearance() -> dict:
    """Assert no blocked source key appears in the flywheel manifest."""
    manifest = json.loads(
        (ROOT / "data" / "flywheel-v2" / "manifest.json").read_text())
    sources = set((manifest.get("sources") or {}).keys())
    bad = {s for s in sources if s.lower().replace("_", "") in
           {b.replace("-", "") for b in BLOCKED_SOURCES}}
    assert not bad, f"blocked sources in mix: {sorted(bad)}"
    return {"flywheel_sources": sorted(sources)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="compose issue-74 mixes")
    parser.add_argument("--out", default="data/aya74")
    args = parser.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    clearance = check_clearance()
    sft_rows, sft_stats = compose_sft()
    dpo_rows, dpo_stats = compose_dpo()

    with open(out / "sft.jsonl", "w") as fh:
        for row in sft_rows:
            fh.write(json.dumps(row) + "\n")
    with open(out / "dpo.jsonl", "w") as fh:
        for row in dpo_rows:
            fh.write(json.dumps(row) + "\n")
    manifest = {
        "issue": 74,
        "seed": SEED,
        "flywheel_revision": "07c4e27919e8ddabbb358dda3c54ea24a6e7dd66",
        "flywheel_sha256": {
            "sft.jsonl": "5b944648ad57ef0483988aa2811d7b0db881e2e8"
                         "aa81352313f486daf91e6424",
            "dpo.jsonl": "a322461981f872989522d76bd7cfde971bc12d5561"
                         "ed255b7d2837d92d136f59",
        },
        "base_model": "CohereLabs/aya-expanse-8b",
        "base_revision": "5062468bf9bc0c6035fd64e06274333ec127d980",
        "sft": sft_stats,
        "dpo": dpo_stats,
        "clearance": clearance,
        "held_out": ["data/redteam/attacks.jsonl",
                     "data/dispatch-pairs/dpo.jsonl"],
        "overlap_disclosure": ("safety (78) and abstention (80) rows are "
                               "trained on for contract teaching; safety "
                               "and abstention eval lanes overlap training "
                               "and are reported as contract checks, while "
                               "red-team, footage, and dispatch-pair lanes "
                               "are held out."),
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"sft={sft_stats} dpo={dpo_stats} -> {out}")


if __name__ == "__main__":
    main()
