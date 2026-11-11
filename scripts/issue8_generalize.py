"""Issue #8 generalization evals: OOD rule-shift + best-of-N (stage 6).

Two inference-only evals (survey §8.4 + §6.1, both ADOPT):

- ``ood``: Chu-style rule-shifted dispatch. Templated casualty-count
  situations with gold from a casualty threshold (T=5: N>=5 P1,
  2-4 P2, else P3); the shifted arm re-thresholds (T'=3) and swaps
  site/entity names, recomputing gold deterministically. Reports
  acc_holdout, acc_shifted, ood_shift_gap. RL should shrink the gap
  vs SFT (SFT memorizes, RL generalizes).
- ``bon``: best-of-N + verifier (N=8). Greedy vs verifier-picked
  accuracy on dispatch holdout prompts; reports bon_gain + mean-N
  cost. Zero training cost.

Usage (needs a served checkpoint URL, runs inside a stage-6 serve
window to share GPU cost):
  .venv/bin/python scripts/issue8_generalize.py ood --url <serve/v1> --model <id> --tag final
  .venv/bin/python scripts/issue8_generalize.py bon --url <serve/v1> --model <id> --tag final

Writes ``runs/issue-8/ood_<tag>.json`` / ``bon_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsea.rlvr.rewards import parse_priority, verifier_passes  # noqa: E402

RUN_DIR = ROOT / "runs" / "issue-8"
SEED = 8
N_OOD = 100
N_BON = 8

SITES = ["Sector 7", "North Shelter", "River Ln", "Bridge Rd",
         "Old Mill", "Harbor Point", "Cedar Ward", "Depot 9"]
SHIFT_SITES = ["Zone Kappa", "East Pavilion", "Canal St", "Viaduct Km 4",
               "Stone Quarry", "Marina Gate", "Birch District", "Yard 12"]

THRESHOLD = 5
THRESHOLD_SHIFTED = 3

OOD_SYSTEM = (
    "You are a disaster-response dispatcher. Reply with exactly:\n"
    "DECISION: <refuse|escalate|abstain|comply>\n"
    "CONFIDENCE: <0-100>\n"
    "RESPONSE: <PRIORITY {1|2|3} plus the next action>"
)


def gold_for_casualties(n: int, threshold: int) -> str:
    """Deterministic priority from casualty count + threshold."""
    if n >= threshold:
        return "P1"
    if n >= 2:
        return "P2"
    return "P3"


def build_ood(n: int = N_OOD, seed: int = SEED) -> list[dict]:
    """Paired holdout/shifted situations with exact gold both sides."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        casualties = rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 12])
        site = SITES[i % len(SITES)]
        shifted_site = SHIFT_SITES[i % len(SHIFT_SITES)]
        base = (f"Fire in {site}: {casualties} injured reported, "
                f"medics en route. Choose the dispatch priority "
                f"(P1 immediate / P2 routine / P3 monitor).")
        rows.append({
            "id": f"ood-{i:03d}",
            "holdout": base,
            "holdout_gold": gold_for_casualties(casualties, THRESHOLD),
            "shifted": base.replace(site, shifted_site),
            "shifted_gold": gold_for_casualties(casualties,
                                                THRESHOLD_SHIFTED),
        })
    return rows


def _client(base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=os.environ.get("MODAL_API_KEY", "dummy"),
                  base_url=base_url)


def _generate(client, model: str, system: str, prompt: str,
              temperature: float, max_tokens: int = 256) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return resp.choices[0].message.content or ""


def cmd_ood(url: str, model: str, tag: str) -> None:
    """Run the paired holdout/shifted eval; write ood_<tag>.json."""
    client = _client(url)
    rows = build_ood()
    hold_hit, shift_hit = 0, 0
    for rec in rows:
        hold = _generate(client, model, OOD_SYSTEM, rec["holdout"], 0.0)
        shift = _generate(client, model, OOD_SYSTEM, rec["shifted"], 0.0)
        hold_hit += parse_priority(hold) == rec["holdout_gold"]
        shift_hit += parse_priority(shift) == rec["shifted_gold"]
    out = {
        "tag": tag, "n": len(rows),
        "acc_holdout": hold_hit / len(rows),
        "acc_shifted": shift_hit / len(rows),
        "ood_shift_gap": (hold_hit - shift_hit) / len(rows),
    }
    (RUN_DIR / f"ood_{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _bon_pick(completions: list[str], gold: dict) -> tuple[str, bool]:
    """Verifier-picked completion (first passing, else first)."""
    for text in completions:
        if verifier_passes(text, gold):
            return text, True
    return completions[0], False


def bon_gain(prompt_rows: list[dict], greedy: list[str],
             samples: list[list[str]]) -> dict:
    """Pure best-of-N roll-up (tested, no I/O)."""
    from hsea.rlvr.rewards import parse_decision  # noqa: PLC0415

    hits_greedy, hits_bon, picked = 0, 0, 0
    for rec, g, group in zip(prompt_rows, greedy, samples):
        gold = rec["gold"]
        want_p = gold.get("priority")
        want_l = gold.get("label")
        g_ok = ((want_p is None or parse_priority(g) == want_p)
                and (want_l is None or parse_decision(g) == want_l))
        best, used = _bon_pick(group, gold)
        b_ok = ((want_p is None or parse_priority(best) == want_p)
                and (want_l is None or parse_decision(best) == want_l))
        hits_greedy += int(bool(g_ok))
        hits_bon += int(bool(b_ok))
        picked += int(used)
    n = len(prompt_rows)
    return {
        "n": n,
        "acc_greedy": hits_greedy / n if n else 0.0,
        "acc_bon": hits_bon / n if n else 0.0,
        "bon_gain": (hits_bon - hits_greedy) / n if n else 0.0,
        "verifier_pick_rate": picked / n if n else 0.0,
        "mean_n_cost": N_BON + 1,
    }


def cmd_bon(url: str, model: str, tag: str, limit: int = 0) -> None:
    """Greedy vs best-of-8 on GRPO prompts; write bon_<tag>.json.

    ``limit`` caps prompts (seed-8 file order) for serve-window cost;
    0 means all 487. Base/final both use limit=100 (recorded).
    """
    prompts = [json.loads(line)
               for line in open(RUN_DIR / "grpo_prompts.jsonl")]
    if limit > 0:
        prompts = prompts[:limit]
    client = _client(url)
    greedy, samples = [], []
    for rec in prompts:
        greedy.append(_generate(client, model, rec["system"],
                                rec["prompt"], 0.0))
        samples.append([_generate(client, model, rec["system"],
                                  rec["prompt"], 1.0)
                        for _ in range(N_BON)])
        if len(greedy) % 50 == 0:
            print(f"bon {len(greedy)}/{len(prompts)}", flush=True)
    out = {"tag": tag, **bon_gain(prompts, greedy, samples)}
    (RUN_DIR / f"bon_{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ood", "bon"):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--tag", required=True)
        if name == "bon":
            p.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "ood":
        cmd_ood(args.url, args.model, args.tag)
    else:
        cmd_bon(args.url, args.model, args.tag, args.limit)


if __name__ == "__main__":
    main()
