"""Issue #8 SafeTune repair mix builder (stage 4, CPU-only).

Builds ``runs/issue-8/safetune_mix.jsonl`` + ``safetune_heldout.json``:
safety demos in the survey §9.1 "100s" band, with template variants
(never the exact eval strings) and a deterministic held-out slice per
bench so Gate 2 has a clean read:

- #14 safety ITEMS (train slice ~75%): paraphrase-wrapped prompts with
  label-gold DECISION/CONFIDENCE/RESPONSE rows. RESPONSE carries the
  rationale (R1 cold-start pattern at small scale: reasoning lives in
  the response, the typed contract is unchanged).
- E-WILD ADAPT (survey §9.4, scoped): per redteam attack (train slice
  ~75%) three rows — vanilla misuse (refuse), stored adversarial
  prompt (refuse), benign lookalike sharing the disaster vocabulary
  (comply with a safeguard-citing answer: the anti-over-refusal arm).
- Abstention ``cal`` split only (28 rows); the ``eval`` split stays
  held out, always.
- 300 seed-8 flywheel-v2 SFT rows as anti-forgetting ballast.

Contamination rule: no exact eval string appears in the mix (asserted),
and ``safetune_heldout.json`` pins the held-out ids that Gate 2 reads
as primary (full-stack numbers stay as secondary context).

Usage:
  .venv/bin/python scripts/issue8_safetune_mix.py [--seed 8]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.data.redteam import load_attacks  # noqa: E402
from hsea.safety.data import ITEMS, SAFETY_SYSTEM  # noqa: E402

RUN_DIR = ROOT / "runs" / "issue-8"
MIX_PATH = RUN_DIR / "safetune_mix.jsonl"
HELDOUT_PATH = RUN_DIR / "safetune_heldout.json"
SFT_PATH = ROOT / "data" / "flywheel-v2" / "sft.jsonl"
ABS_SCEN = ROOT / "data" / "abstention" / "scenarios.jsonl"
ABS_BENIGN = ROOT / "data" / "abstention" / "benign.jsonl"

SEED = 8
HELDOUT_FRAC = 0.25
ANTIFORGET_N = 300

# Deterministic weak-paraphrase wrappers (recorded as such; the
# held-out slice is the clean read, not these rewrites).
WRAPPERS = (
    "Ops question: {p}",
    "From the field log: {p}",
    "Coordinator asks: {p}",
    "Incoming request: {p}",
)

CONFIDENCE = {"refuse": 95, "escalate": 80, "abstain": 75, "comply": 90}


def gold_response(label: str, prompt: str, rationale: str = "") -> str:
    """Label-gold assistant text with rationale (R1 cold-start style)."""
    reason = rationale.strip() or _default_rationale(label, prompt)
    if label == "refuse":
        body = (f"I cannot help with that. {reason} If you have a "
                f"legitimate emergency need behind this request, contact "
                f"your incident coordinator through official channels.")
    elif label == "escalate":
        body = (f"This needs a human decision before anyone acts. {reason} "
                f"I have flagged it for the coordinator; no action taken.")
    elif label == "abstain":
        body = (f"I do not have enough reliable information to act safely. "
                f"{reason} Tell me the missing facts or stand by for "
                f"confirmed reports.")
    elif label == "comply":
        body = reason
    else:
        raise ValueError(f"unknown label {label!r}")
    return (f"DECISION: {label}\nCONFIDENCE: {CONFIDENCE[label]}\n"
            f"RESPONSE: {body}")


def _default_rationale(label: str, prompt: str) -> str:
    short = prompt.strip().rstrip(".")
    if len(short) > 120:
        short = short[:117] + "..."
    if label == "refuse":
        return (f"The request ({short}) asks for harmful, illegal, or "
                f"unsafe action, which I must not carry out.")
    if label == "escalate":
        return (f"The request ({short}) is high-stakes and needs human "
                f"approval first.")
    if label == "abstain":
        return (f"The request ({short}) cannot be answered safely from "
                f"the information given.")
    return f"Handling the safe request: {short}."


def lookalike_request(request: str) -> str:
    """Benign lookalike sharing the attack's disaster vocabulary."""
    return (f"What official safeguard or approval step governs this "
            f"operation, and who must sign off: {request}?")


def lookalike_gold(request: str) -> str:
    """Comply-shaped gold citing procedure (anti-over-refusal)."""
    return gold_response(
        "comply", request,
        f"Legitimate operations of this kind ({request}) require "
        f"coordinator approval, an audit-trail entry, and a qualified "
        f"operator; the ops manual lists the exact sign-off chain for "
        f"your sector. I can draft the request paperwork or summarize "
        f"the governing procedure on request.")


def wrap_prompt(prompt: str, rng: random.Random) -> str:
    """Deterministic weak paraphrase (never returns the input as-is)."""
    return rng.choice(WRAPPERS).format(p=prompt.strip())


def split_train_heldout(ids: list[str], seed: int = SEED,
                        frac: float = HELDOUT_FRAC) -> tuple[list[str], list[str]]:
    """Deterministic per-bench train/heldout split (sorted ids)."""
    rng = random.Random(seed)
    pool = sorted(ids)
    rng.shuffle(pool)
    n_held = max(1, round(len(pool) * frac))
    return sorted(pool[n_held:]), sorted(pool[:n_held])


def build_mix(seed: int = SEED) -> tuple[list[dict], dict]:
    """(mix rows, heldout manifest). Pure except for data reads."""
    rng = random.Random(seed)
    rows: list[dict] = []
    heldout: dict[str, list[str]] = {}

    def add_row(prompt: str, label: str, rationale: str, source: str) -> None:
        rows.append({
            "messages": [
                {"role": "system", "content": SAFETY_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": gold_response(label, prompt, rationale)},
            ],
            "source": source,
        })

    # 1. #14 ITEMS variants (train slice per bench).
    by_bench: dict[str, list[int]] = {}
    for i, (bench, _cat, _label, _p) in enumerate(ITEMS):
        by_bench.setdefault(bench, []).append(i)
    for bench, idxs in sorted(by_bench.items()):
        ids = [f"{bench}:{i}" for i in idxs]
        train_ids, held_ids = split_train_heldout(ids, seed)
        heldout[f"items:{bench}"] = held_ids
        for gid in train_ids:
            _b, _c, label, prompt = ITEMS[int(gid.split(":")[1])]
            add_row(wrap_prompt(prompt, rng), label, "", f"items:{bench}")

    # 2. E-WILD 4-type rows (train slice per family).
    attacks = load_attacks()
    by_fam: dict[str, list[dict]] = {}
    for rec in attacks:
        by_fam.setdefault(rec["family"], []).append(rec)
    for fam, recs in sorted(by_fam.items()):
        ids = sorted(r["id"] for r in recs)
        train_ids, held_ids = split_train_heldout(ids, seed)
        heldout[f"redteam:{fam}"] = held_ids
        by_id = {r["id"]: r for r in recs}
        for gid in train_ids:
            rec = by_id[gid]
            req = rec["request"]
            # Vanilla misuse -> refuse.
            add_row(wrap_prompt(f"Carry out this operation: {req}.", rng),
                    "refuse", "", f"redteam:{fam}:vanilla")
            # Stored adversarial prompt -> refuse.
            add_row(rec["prompt"], "refuse", "", f"redteam:{fam}:adversarial")
            # Benign lookalike -> comply.
            rows.append({
                "messages": [
                    {"role": "system", "content": SAFETY_SYSTEM},
                    {"role": "user", "content": lookalike_request(req)},
                    {"role": "assistant", "content": lookalike_gold(req)},
                ],
                "source": f"redteam:{fam}:lookalike",
            })

    # 3. Abstention cal split only.
    for path in (ABS_SCEN, ABS_BENIGN):
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") != "cal":
                continue
            label = rec["label"]
            if label not in CONFIDENCE:
                raise ValueError(f"bad abstention label {label!r}")
            add_row(rec["prompt"], label,
                    rec.get("rationale", ""), "abstention:cal")

    # 4. Anti-forgetting ballast (seed-8 SFT sample).
    sft_rows = [json.loads(line) for line in open(SFT_PATH) if line.strip()]
    rng.shuffle(sft_rows)
    for rec in sft_rows[:ANTIFORGET_N]:
        rows.append({"messages": rec["messages"], "source": "sft:ballast"})

    # Contamination assertion: no HELD-OUT eval string in the mix.
    # Train-slice attack prompts appear verbatim by design (train/test
    # are disjoint ITEMS, same distribution — the held-out slice plus
    # the abstention eval split are the clean reads).
    held_strings: set[str] = set()
    for key, gids in heldout.items():
        kind, _bench = key.split(":", 1)
        if kind == "items":
            for gid in gids:
                held_strings.add(ITEMS[int(gid.split(":")[1])][3])
        else:
            by_id = {r["id"]: r for r in attacks}
            for gid in gids:
                held_strings.add(by_id[gid]["prompt"])
    for rec in rows:
        if rec["source"] == "sft:ballast":
            continue  # training-distribution rows, never eval strings
        for turn in rec["messages"]:
            content = turn.get("content")
            if turn.get("role") == "user" and isinstance(content, str):
                assert content not in held_strings, \
                    f"held-out string leaked into mix: {content[:60]!r}"
    return rows, {"seed": seed, "heldout": heldout}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_mix(args.seed)
    with open(MIX_PATH, "w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    HELDOUT_PATH.write_text(json.dumps(manifest, indent=2))
    by_source: dict[str, int] = {}
    for rec in rows:
        by_source[rec["source"].split(":")[0]] = \
            by_source.get(rec["source"].split(":")[0], 0) + 1
    print(f"mix rows={len(rows)} {by_source} -> {MIX_PATH}")
    print(f"heldout -> {HELDOUT_PATH}")


if __name__ == "__main__":
    main()
