"""Fixed-seed replay bundles: before/after pairs anyone can re-score.

Builds ``runs/issue-<N>/replays/`` from committed eval artifacts: one
row per judged prompt with the input, per-stage generations, every
judge's reasoning + verdict, and the seed. ``--check`` re-derives the
headline numbers (dual agreement, T5 wins) from the bundle alone and
fails when they drift from the recorded analysis.

Usage:
    python scripts/replay_loop.py --issue 74
    python scripts/replay_loop.py --issue 74 --check
    python scripts/replay_loop.py --issue 74 --upload  # HF dataset

Only issue 74 is implemented (extend ISSUES for the next loop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ISSUES = {74: {"seed": 74, "run_dir": "runs/aya74",
               "checkpoints": ("base", "sft", "dpo")}}
GPU_FINETUNES = ("repair",)


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_bundle(issue: int) -> dict:
    """Assemble the replay bundle from committed artifacts."""
    from aya74_eval import _judge_plan
    from hsea.judge.templates import (parse_judgment, parse_pairwise,
                                      ABSTENTION_LABELS, REDTEAM_LABELS,
                                      SAFETY_LABELS)

    cfg = ISSUES[issue]
    run = ROOT / cfg["run_dir"]
    prompts = {json.loads(line)["key"]: json.loads(line)
               for line in open(run / "eval_prompts.jsonl")}
    comps = {}
    for ckpt in cfg["checkpoints"]:
        comps[ckpt] = {}
        for line in open(run / f"eval_{ckpt}.jsonl"):
            rec = json.loads(line)
            comps[ckpt][rec["key"]] = rec.get("completion", "")
    hf: dict[str, dict[str, dict]] = {}
    for line in open(run / "judgments.jsonl"):
        row = json.loads(line)
        hf.setdefault(row["key"], {})[row["judge"]] = row
    meta: dict[str, dict] = {}
    for line in open(run / "judgments_meta.jsonl"):
        row = json.loads(line)
        if "error" not in row:
            meta[row["key"]] = row

    labels = {"T2": SAFETY_LABELS, "T3": ABSTENTION_LABELS,
              "T4": REDTEAM_LABELS, "T1": ("correct", "close", "wrong")}
    plan = _judge_plan()
    rows = []
    for key, kind, system, user in plan:
        if kind == "T5":
            _, order, pid = key.split(":", 2)
            p = prompts[pid]
            sides = {"base": comps["base"].get(pid, ""),
                     "dpo": comps["dpo"].get(pid, "")}
            ordered = [sides["base"], sides["dpo"]] if order == "orig" \
                else [sides["dpo"], sides["base"]]
            row = {"key": key, "kind": kind, "seed": cfg["seed"],
                   "input": {"situation": p["prompt"], "order": order,
                             "side_a": ordered[0], "side_b": ordered[1]},
                   "generations": {**sides,
                                   "sft": comps["sft"].get(pid, "")}}
        else:
            _, ckpt, pid = key.split(":", 2)
            p = prompts[pid]
            row = {"key": key, "kind": kind, "seed": cfg["seed"],
                   "input": {"system": system, "user": user,
                             "gold": p.get("gold", ""),
                             "prompt_key": pid,
                             "judged_checkpoint": ckpt},
                   "generations": {c: comps[c].get(pid, "")
                                   for c in cfg["checkpoints"]}}
        judgments = {}
        for judge, rec in list(hf.get(key, {}).items()) + \
                ([("meta-ember", meta[key])] if key in meta else []):
            text = rec["text"]
            parsed = parse_pairwise(text) if kind == "T5" \
                else parse_judgment(text, labels[kind])
            judgments[judge] = {
                "model": rec["model"], "text": text,
                "reasoning": parsed.get("reasoning", ""),
                "verdict": parsed.get("verdict", ""),
                "score": parsed.get("score", ""),
                "prompt_tokens": rec.get("prompt_tokens", 0),
                "completion_tokens": rec.get("completion_tokens", 0),
            }
        row["judgments"] = judgments
        rows.append(row)

    out_dir = ROOT / f"runs/issue-{issue}" / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"aya{issue}_replay.jsonl"
    with open(bundle, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    manifest = {
        "issue": issue, "seed": cfg["seed"], "rows": len(rows),
        "judges": sorted({j for r in rows for j in r["judgments"]}),
        "sources": {name: _sha16(run / name) for name in
                    ("eval_prompts.jsonl", "eval_base.jsonl",
                     "eval_sft.jsonl", "eval_dpo.jsonl",
                     "judgments.jsonl", "judgments_meta.jsonl")},
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"bundle: {len(rows)} rows -> {bundle}")
    return manifest


def build_gpu_bundle(issue: int,
                     finetunes: tuple[str, ...] = GPU_FINETUNES) -> dict:
    """Assemble the GPU-leg replay bundle (post-repair T5 duels).

    One row per judged T5 prompt per finetune: input, base/finetune
    generations, and every judge's reasoning + verdict. Keys are
    namespaced ``T5:<order>:<finetune>:<pid>`` so they never collide
    with the Phase-1 bundle.
    """
    from hsea.judge.templates import parse_pairwise

    cfg = ISSUES[issue]
    run = ROOT / cfg["run_dir"]
    prompts = {json.loads(line)["key"]: json.loads(line)
               for line in open(run / "eval_prompts.jsonl")}
    comps: dict[str, dict[str, str]] = {}
    for ckpt in ("base",) + tuple(finetunes):
        comps[ckpt] = {}
        for line in open(run / f"eval_{ckpt}.jsonl"):
            rec = json.loads(line)
            comps[ckpt][rec["key"]] = rec.get("completion", "")
    rows = []
    sources: dict[str, str] = {
        "eval_prompts.jsonl": _sha16(run / "eval_prompts.jsonl"),
        "eval_base.jsonl": _sha16(run / "eval_base.jsonl"),
    }
    for ft in finetunes:
        sources[f"eval_{ft}.jsonl"] = _sha16(run / f"eval_{ft}.jsonl")
        hf: dict[str, dict[str, dict]] = {}
        for line in open(run / f"judgments_t5_{ft}.jsonl"):
            row = json.loads(line)
            hf.setdefault(row["key"], {})[row["judge"]] = row
        sources[f"judgments_t5_{ft}.jsonl"] = \
            _sha16(run / f"judgments_t5_{ft}.jsonl")
        meta: dict[str, dict] = {}
        for line in open(run / f"judgments_meta_t5_{ft}.jsonl"):
            row = json.loads(line)
            if "error" not in row:
                meta[row["key"]] = row
        sources[f"judgments_meta_t5_{ft}.jsonl"] = \
            _sha16(run / f"judgments_meta_t5_{ft}.jsonl")
        for key in sorted(hf):
            _, order, pid = key.split(":", 2)
            p = prompts[pid]
            sides = {"base": comps["base"].get(pid, ""),
                     ft: comps[ft].get(pid, "")}
            ordered = [sides["base"], sides[ft]] if order == "orig" \
                else [sides[ft], sides["base"]]
            judgments = {}
            for judge, rec in list(hf[key].items()) + \
                    ([("meta-ember", meta[key])] if key in meta else []):
                parsed = parse_pairwise(rec["text"])
                judgments[judge] = {
                    "model": rec["model"], "text": rec["text"],
                    "reasoning": parsed.get("reasoning", ""),
                    "verdict": parsed.get("verdict", ""),
                    "score": parsed.get("score", ""),
                    "prompt_tokens": rec.get("prompt_tokens", 0),
                    "completion_tokens": rec.get("completion_tokens", 0),
                }
            rows.append({"key": f"T5:{order}:{ft}:{pid}", "kind": "T5",
                         "seed": cfg["seed"], "finetune": ft,
                         "input": {"situation": p["prompt"], "order": order,
                                   "side_a": ordered[0],
                                   "side_b": ordered[1]},
                         "generations": sides, "judgments": judgments})
    out_dir = ROOT / f"runs/issue-{issue}" / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"aya{issue}_gpu_replay.jsonl"
    with open(bundle, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    manifest = {
        "issue": issue, "seed": cfg["seed"], "rows": len(rows),
        "finetunes": list(finetunes),
        "judges": sorted({j for r in rows for j in r["judgments"]}),
        "sources": sources,
    }
    with open(out_dir / "manifest_gpu.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"gpu bundle: {len(rows)} rows -> {bundle}")
    return manifest


def check_gpu_bundle(issue: int,
                     finetunes: tuple[str, ...] = GPU_FINETUNES) -> None:
    """Re-derive GPU-leg T5 wins from the bundle; fail on drift."""
    from hsea.judge.templates import parse_pairwise

    run = ROOT / ISSUES[issue]["run_dir"]
    out_dir = ROOT / f"runs/issue-{issue}" / "replays"
    rows = [json.loads(line)
            for line in open(out_dir / f"aya{issue}_gpu_replay.jsonl")]
    for ft in finetunes:
        ft_rows = {r["key"]: r for r in rows if r.get("finetune") == ft}
        assert ft_rows, f"no gpu replay rows for {ft}"
        spend = json.load(open(run / f"judge_spend_t5_{ft}.json"))
        analysis = json.load(open(run / f"analysis_t5_{ft}.json"))
        agree3 = json.load(open(run / f"agreement3_t5_{ft}.json"))

        def wins(judge: str) -> dict:
            tally = {ft: 0, "base": 0, "tie": 0, "inconsistent": 0}
            pairs: dict[str, dict[str, str]] = {}
            for k in ft_rows:
                _, order, _, pid = k.split(":", 3)
                pairs.setdefault(pid, {})[order] = parse_pairwise(
                    ft_rows[k]["judgments"][judge]["text"])["verdict"] or ""
            for orders in pairs.values():
                o, s = orders.get("orig", ""), orders.get("swap", "")
                flip = {"A": "B", "B": "A", "C": "C"}
                if o not in flip or s != flip[o]:
                    tally["inconsistent"] += 1
                elif o == "C":
                    tally["tie"] += 1
                elif o == "B":
                    tally[ft] += 1
                else:
                    tally["base"] += 1
            return tally

        for judge in spend["judges"]:
            want = {k: analysis["t5"][judge][k]
                    for k in (ft, "base", "tie", "inconsistent")}
            got = wins(judge)
            assert got == want, f"T5/{ft} {judge}: {got} != {want}"
            print(f"T5/{ft} {judge}: {got} matches")
        wantm = {k: agree3["t5"]["meta-ember"][k]
                 for k in (ft, "base", "tie", "inconsistent")}
        gotm = wins("meta-ember")
        assert gotm == wantm, f"T5/{ft} meta: {gotm} != {wantm}"
        print(f"T5/{ft} meta-ember: {gotm} matches")


def check_bundle(issue: int) -> None:
    """Re-derive headline numbers from the bundle; fail on drift."""
    from hsea.judge.templates import (parse_judgment, parse_pairwise,
                                      ABSTENTION_LABELS, REDTEAM_LABELS,
                                      SAFETY_LABELS)

    labels = {"T2": SAFETY_LABELS, "T3": ABSTENTION_LABELS,
              "T4": REDTEAM_LABELS, "T1": ("correct", "close", "wrong")}

    cfg = ISSUES[issue]
    run = ROOT / cfg["run_dir"]
    out_dir = ROOT / f"runs/issue-{issue}" / "replays"
    rows = [json.loads(line)
            for line in open(out_dir / f"aya{issue}_replay.jsonl")]
    by_key = {r["key"]: r for r in rows}

    analysis = json.load(open(run / "analysis.json"))
    for kind in ("T2", "T3", "T4", "T1"):
        keys = sorted(k for k in by_key if k.startswith(kind + ":"))

        def verdicts(judge: str) -> list[str]:
            out = []
            for k in keys:
                text = by_key[k]["judgments"][judge]["text"]
                out.append(parse_judgment(text, labels[kind])["verdict"]
                           or "none")
            return out

        judges = analysis["judges"]
        va, vb = verdicts(judges[0]), verdicts(judges[1])
        agree = sum(1 for x, y in zip(va, vb) if x == y)
        want = analysis["single"][kind]["dual_agree"]
        assert agree == want, f"{kind}: bundle {agree} != recorded {want}"
        print(f"{kind}: dual agree {agree}/{len(keys)} matches")

    def wins(judge: str) -> dict:
        tally = {"dpo": 0, "base": 0, "tie": 0, "inconsistent": 0}
        pairs: dict[str, dict[str, str]] = {}
        for k in by_key:
            if not k.startswith("T5:"):
                continue
            _, order, pid = k.split(":", 2)
            pairs.setdefault(pid, {})[order] = parse_pairwise(
                by_key[k]["judgments"][judge]["text"])["verdict"] or ""
        for orders in pairs.values():
            o, s = orders.get("orig", ""), orders.get("swap", "")
            flip = {"A": "B", "B": "A", "C": "C"}
            if o not in flip or s != flip[o]:
                tally["inconsistent"] += 1
            elif o == "C":
                tally["tie"] += 1
            elif o == "B":
                tally["dpo"] += 1
            else:
                tally["base"] += 1
        return tally

    want5 = {k: analysis["t5"][k]
             for k in ("dpo", "base", "tie", "inconsistent")}
    got5 = wins(analysis["judges"][0])
    assert got5 == want5, f"T5 hf: {got5} != {want5}"
    print(f"T5 {analysis['judges'][0]}: {got5} matches")
    agree3 = json.load(open(run / "agreement3.json"))
    wantm = {k: agree3["t5"]["meta-ember"][k]
             for k in ("dpo", "base", "tie", "inconsistent")}
    gotm = wins("meta-ember")
    assert gotm == wantm, f"T5 meta: {gotm} != {wantm}"
    print(f"T5 meta-ember: {gotm} matches")
    check_gpu_bundle(issue)
    print("REPLAY CHECK OK")


def upload_bundle(issue: int) -> None:
    """Upload the bundle + manifest to the issue's HF dataset repo."""
    from huggingface_hub import HfApi

    repo = f"sanggusti/aya{issue}-replays"
    out_dir = ROOT / f"runs/issue-{issue}" / "replays"
    token = (Path.home() / ".cache" / "huggingface" / "token") \
        .read_text().strip()
    api = HfApi(token=token)
    print(f"hub user: {api.whoami()['name']}")
    card = f"""---
license: cc-by-nc-4.0
tags: [replay, eval, disaster-response, hsea]
---
# Issue #{issue} replay bundle (fixed seed {ISSUES[issue]['seed']})

Before/after example pairs: inputs, per-stage generations
(base/sft/dpo), and every judge's reasoning + verdicts.
`aya{issue}_gpu_replay.jsonl` extends the bundle with the GPU-leg
post-repair T5 duels (base/repair generations + all three judges).
Reproduced by `scripts/replay_loop.py --issue {issue} --check`
in sanggusti/expert-spork. Research-only (NC stack).
"""
    (out_dir / "README.md").write_text(card)
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=False)
    api.upload_folder(repo_id=repo, repo_type="dataset",
                      folder_path=str(out_dir))
    print(f"{repo}: uploaded")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="replay bundles")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    if args.issue not in ISSUES:
        raise SystemExit(f"unsupported issue {args.issue}")
    if args.upload:
        upload_bundle(args.issue)
    elif args.check:
        check_bundle(args.issue)
    else:
        build_bundle(args.issue)
        build_gpu_bundle(args.issue)


if __name__ == "__main__":
    main()
