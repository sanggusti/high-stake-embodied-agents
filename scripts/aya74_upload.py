"""Upload issue-74 adapters + logs to the Hub (one repo per stage).

Prerequisites (run first)::
    modal volume get dmb-flywheel-models aya74/sft runs/aya74/dl/sft
    modal volume get dmb-flywheel-models aya74/dpo runs/aya74/dl/dpo
    modal volume get dmb-flywheel-models aya74/logs runs/aya74/dl/logs

Usage:
    python scripts/aya74_upload.py --dir runs/aya74/dl [--dry-run]

Creates (or reuses, owned by this issue) ``sanggusti/aya-expanse-8b-hsea-sft``
and ``sanggusti/aya-expanse-8b-hsea-dpo``: adapter weights + tokenizer +
trainer state + model card. Never prints the HF token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPOS = {
    "sft": "sanggusti/aya-expanse-8b-hsea-sft",
    "dpo": "sanggusti/aya-expanse-8b-hsea-dpo",
    "repair": "sanggusti/aya-expanse-8b-hsea-repair",
    "kto": "sanggusti/aya-expanse-8b-hsea-kto",
    "ipo": "sanggusti/aya-expanse-8b-hsea-ipo",
    "rft": "sanggusti/aya-expanse-8b-hsea-rft",
    "grpo": "sanggusti/aya-expanse-8b-hsea-grpo",
}
LICENSE = "cc-by-nc-4.0"


def _model_card(stage: str, manifest: dict) -> str:
    base = manifest["base_model"]
    rev = manifest["base_revision"]
    data_rev = manifest["flywheel_revision"]
    if stage == "sft":
        desc = ("LoRA SFT adapter (r16/alpha32, 3 epochs, eff batch 8, "
                "LR 2e-4 cosine) on the issue-74 mix: flywheel SFT 8,673 "
                "+ HSEA safety/abstention contract demos.")
        rows = manifest["sft"]["total"]
    elif stage == "dpo":
        desc = ("LoRA DPO adapter (beta 0.1, 1 epoch, eff batch 8, LR "
                "5e-5 cosine) over the merged SFT checkpoint on the "
                "issue-74 DPO mix: flywheel dispatch pairs 7,000 + HSEA "
                "safety pairs. Reference: merged SFT (frozen).")
        rows = manifest["dpo"]["total"]
    elif stage == "repair":
        desc = ("LoRA repair adapter (Stage 4 SafeTune re-tune, 3 epochs "
                "via entrypoint default, eff batch 8, LR 5e-5) over "
                "merged DPO + fresh LoRA on repair_sft.jsonl (438 rows: "
                "unified-footer safety/abstention, dispatch no-footer "
                "anchors, WJ-method mix).")
        rows = 438
    elif stage == "kto":
        desc = ("LoRA KTO adapter (beta 0.1, 1 epoch, eff batch 8, LR "
                "5e-5) over merged DPO + fresh LoRA on kto.jsonl "
                "(160 binary abstention labels).")
        rows = 160
    elif stage == "ipo":
        desc = ("LoRA IPO adapter (loss_type ipo, beta 0.1, 1 epoch, "
                "eff batch 8, LR 5e-5) over merged SFT on the 578 "
                "debiased repair DPO pairs (bake-off pilot).")
        rows = 578
    elif stage == "rft":
        desc = ("LoRA RFT baseline adapter (SFT 1 epoch, eff batch 8, "
                "LR 5e-5) over merged DPO on 38 verifier-kept dispatch "
                "samples (100 prompts x 4 at temp 0.7, rule-reward "
                "at least 1.0).")
        rows = 38
    else:
        desc = ("LoRA GRPO pilot adapter (100 steps, 4 generations, "
                "batch 2x2, LR 1e-5, beta 0.04) cold-started from "
                "merged repair on 198 stratified dispatch prompts with "
                "SWE-RL field-match + R1 rule rewards.")
        rows = 198
    return f"""---
base_model: {base}
base_model_revision: {rev}
license: {LICENSE}
tags: [lora, dpo, disaster-response, hsea]
---

# Aya Expanse 8B HSEA {stage.upper()} adapter (issue #74)

{desc}

- Base: `{base}` @ `{rev}` (CC-BY-NC-4.0, research-only)
- Data: `sanggusti/dmb-flywheel-v2` @ `{data_rev}` ({rows} rows)
- Recipe: shared model-comparison-protocol ( rank/alpha/targets/epochs
  identical across #8/#73/#74; deviations in the issue report)
- Eval: base-vs-finetune battery + T1-T5 judges, see
  `docs/reports/issue-74-aya-expanse.md` in sanggusti/expert-spork.
- Research-only: NC-licensed base and mix; not for product use.
"""


def main(argv: list[str] | None = None) -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description="upload #74 adapters")
    parser.add_argument("--dir", default="runs/aya74/dl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", default="",
                        help="comma-separated subset of stages "
                             "(default: all)")
    args = parser.parse_args(argv)
    dl = ROOT / args.dir
    manifest = json.loads((ROOT / "data" / "aya74" / "manifest.json").read_text())

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    token = token_file.read_text().strip()
    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"hub user: {who}")

    want = [s for s in args.stages.split(",") if s] if args.stages else None
    for stage, repo in REPOS.items():
        if want is not None and stage not in want:
            continue
        src = dl / stage
        assert src.is_dir(), f"missing {src} (modal volume get first)"
        card = _model_card(stage, manifest)
        (src / "README.md").write_text(card)
        log = dl / "logs" / f"{stage}_trainer_state.json"
        if log.exists():
            (src / "trainer_state.json").write_text(log.read_text())
        print(f"{repo}: {len(list(src.iterdir()))} files ready in {src}")
        if args.dry_run:
            continue
        api.create_repo(repo, exist_ok=True, private=False)
        api.upload_folder(repo_id=repo, folder_path=str(src))
        print(f"{repo}: uploaded")


if __name__ == "__main__":
    main()
