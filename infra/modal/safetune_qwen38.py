"""SafeTune repair: safety-mix re-tune over the DPO checkpoint (issue #8).

Stage 4 of the full loop. Continues QLoRA SFT from the stage-2 DPO
adapter on ``runs/issue-8/safetune_mix.jsonl`` (459 rows: #14 ITEMS
variants, E-WILD disaster-misuse + lookalike rows, abstention ``cal``,
300 SFT ballast rows; see scripts/issue8_safetune_mix.py). Runs until
Gate 2 passes (agentharm misuse refusal 1.0, benign >= 0.85,
abstention >= 0.80), re-running the safety stack after the repair.

Estimate-first without a separate probe: this run reuses the stage-1
measured s/it (same model, same max_seq, same batch geometry), so the
pre-run estimate IS the extrapolation (see ESTIMATE_S_PER_IT_SRC).
A ``--mode probe`` exists for API symmetry but the plan records the
SFT-derived estimate, not a second paid probe.

Same binding rules as stage 1 (see finetune_qwen38.py docstring):
pinned lock releases, prebuilt-wheels-only, GPU fallback chain,
sequence vs #73/#74.

Usage (unsandboxed shell; GPU spend applies):
  modal volume put dmb-flywheel runs/issue-8/safetune_mix.jsonl /flywheel-v2/safetune_mix.jsonl
  modal run infra/modal/safetune_qwen38.py --mode train
RFT reuse (stage 5 baseline; same code path, overridden IO):
  modal volume put dmb-flywheel runs/issue-8/rft_filtered.jsonl /flywheel-v2/rft_filtered.jsonl
  modal run infra/modal/safetune_qwen38.py --mode train \
    --dataset /data/flywheel-v2/rft_filtered.jsonl \
    --parent /models/qwen38-flywheel-v2-safetune \
    --out /models/qwen38-flywheel-v2-rft \
    --repo sanggusti/qwen38-flywheel-v2-rft --epochs 1

Cost estimate (pre-GPU, from the stage-1 probe anchor): 174 steps
(459 rows / eff-batch 8 x 3 epochs) at ~7 s/it = ~20 min + ~30 min
setup on A100-80GB -> ~$2.10.
"""

from __future__ import annotations

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
DATA_REPO = "sanggusti/dmb-flywheel-v2"
DATA_REVISION = "b44237e145af1f0d9cfa3f22ec36f77d22ca5393"
DPO_ADAPTER_DIR = "/models/qwen38-flywheel-v2-dpo"
HF_REPO = "sanggusti/qwen38-flywheel-v2-safetune"

DATASET_PATH = "/data/flywheel-v2/safetune_mix.jsonl"
ADAPTER_DIR = "/models/qwen38-flywheel-v2-safetune"
OUTPUT_DIR = "/models/qwen38-flywheel-v2-safetune-merged"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 2048
SEED = 8

PIN_TRANSFORMERS = "transformers==5.22.2"
PIN_TRL = "trl==1.7.1"
PIN_TORCH = "torch==2.10.0"  # cu128: newest torch with a prebuilt cu12 causal-conv1d wheel
PIN_PEFT = "peft==0.21.2"
PIN_DATASETS = "datasets==5.1.0"
PIN_ACCELERATE = "accelerate==1.15.0"
PIN_BITSANDBYTES = "bitsandbytes==0.50.2"
PIN_HF_HUB = "huggingface-hub==1.33.0"

GPU_CHAIN = ["A100-80GB", "H100"]
GPU_RATES = {"A100-80GB": 2.49, "H100": 3.95, "H200": 4.54}

# Reuses the stage-1 measured s/it (same model/geometry); the recorded
# value replaces this anchor before the run commits.
ESTIMATE_S_PER_IT = 7.0
PROBE_STEPS = 50
ABORT_FACTOR = 2.0
STAGE_BUDGET_USD = 3.0


ABORT_CHECK_STEPS = 50


def should_abort(elapsed_s: float, steps: int,
                 anchor_s_per_it: float = ESTIMATE_S_PER_IT,
                 factor: float = ABORT_FACTOR) -> dict:
    """In-run 2x gate (pure): abort a full run whose early steps run
    slower than factor x anchor. Same binding rule as the probe, zero
    extra setup cost (checked inside the run at ABORT_CHECK_STEPS)."""
    s_per_it = elapsed_s / max(1, steps)
    over = s_per_it > anchor_s_per_it * factor
    return {"s_per_it": round(s_per_it, 2), "abort": over}

def probe_verdict(s_per_it: float, cost_usd: float | None = None,
                  budget_usd: float | None = None) -> dict:
    """GO / ABORT / NO-GO from the probe: the binding 2x time rule
    plus the stage cost gate (NO-GO when the extrapolated cost
    exceeds the stage budget even if time is GO)."""
    over = s_per_it > ESTIMATE_S_PER_IT * ABORT_FACTOR
    over_budget = (cost_usd is not None and budget_usd is not None
                   and cost_usd > budget_usd)
    verdict = "ABORT" if over else ("NO-GO" if over_budget else "GO")
    return {
        "measured_s_per_it": round(s_per_it, 2),
        "anchor_s_per_it": ESTIMATE_S_PER_IT,
        "factor": round(s_per_it / ESTIMATE_S_PER_IT, 2),
        "cost_usd": cost_usd,
        "budget_usd": budget_usd,
        "verdict": verdict,
    }


def detect_fused() -> str:
    """Prebuilt-kernel path report (fused / PARTIAL / reference)."""
    try:
        import causal_conv1d  # noqa: PLC0415

        cc1d = causal_conv1d.__version__
    except ImportError:
        cc1d = None
    try:
        import fla  # noqa: PLC0415

        fla_v = getattr(fla, "__version__", "?")
    except ImportError:
        fla_v = None
    if cc1d and fla_v:
        return f"fused prebuilt (causal-conv1d {cc1d}, fla {fla_v})"
    if cc1d or fla_v:
        return f"PARTIAL kernels (cc1d={cc1d}, fla={fla_v})"
    return "reference impl (no prebuilt kernels)"


def classify_gpu(gpu_name: str) -> str:
    """nvidia-smi product name -> ledger GPU class (H200 is distinct
    from H100: different rate, different placement pool)."""
    if "H200" in gpu_name:
        return "H200"
    if "H100" in gpu_name:
        return "H100"
    if "A100" in gpu_name:
        return "A100-80GB"
    return "A100-80GB"


def normalize_content(content) -> str:
    """Flatten one message content to a plain string (stage-1 salvage)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"unsupported content type {type(content).__name__}")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            raise ValueError(f"unsupported content block {str(block)[:80]!r}")
    return "".join(parts)


def coerce_record(record: dict) -> dict:
    """Normalize one mix row to ``{"messages", "source"}``."""
    if not isinstance(record, dict) or "messages" not in record:
        raise ValueError("record lacks a messages list")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out = []
    for turn in messages:
        if not isinstance(turn, dict):
            raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
        role = turn.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown role {role!r}")
        out.append({"role": role, "content": normalize_content(
            turn.get("content", ""))})
    roles = [t["role"] for t in out]
    if "assistant" not in roles:
        raise ValueError("no assistant turn to train on")
    return {"messages": out, "source": str(record.get("source", "?"))}


def count_steps(n_rows: int, batch: int, accum: int, epochs: int) -> int:
    """Honest step math: ceil(rows / eff-batch) per epoch x epochs."""
    import math

    eff = batch * accum
    return math.ceil(n_rows / eff) * epochs


def extrapolate_cost(steps: int, s_per_it: float, gpu: str,
                     setup_minutes: float = 30.0) -> dict:
    """Wall time + cost extrapolation from a measured s/it."""
    rate = GPU_RATES.get(gpu, GPU_RATES["A100-80GB"])
    hours = steps * s_per_it / 3600 + setup_minutes / 60
    return {
        "steps": steps,
        "s_per_it": round(s_per_it, 2),
        "gpu": gpu,
        "hours": round(hours, 2),
        "cost_usd": round(hours * rate, 2),
        "rate_usd_hr": rate,
    }


def build_model_card(train_loss: float, steps: int, rows: int,
                     commit: str, gpu: str) -> str:
    """SafeTune adapter model card (DPO parent, mix, research gate)."""
    return f"""---
license: other
license_name: mixed-per-source-research-only
base_model: {BASE_MODEL}
base_model_revision: {BASE_REVISION}
tags:
- lora
- disaster-response
- safetune
---

# Qwen3.8-27B flywheel-v2 SafeTune (issue #8)

Safety-mix repair over the stage-2 DPO adapter: {rows} rows (#14 ITEMS
variants, E-WILD disaster-misuse + benign lookalikes, abstention cal,
SFT ballast). Research-only: gated like its data.

## Config

- parent: stage-2 DPO adapter ({DPO_ADAPTER_DIR})
- method: QLoRA NF4 continued (r={LORA_R} alpha={LORA_ALPHA})
- epochs={EPOCHS} batch={BATCH_SIZE} accum={GRAD_ACCUM} lr={LEARNING_RATE}
- max_seq={MAX_SEQ_LENGTH} seed={SEED}
- pins: {PIN_TRANSFORMERS} {PIN_TRL} {PIN_TORCH} {PIN_PEFT}
- gpu: {gpu} (fallback chain {GPU_CHAIN})
- code commit: {commit}

## Result

- train_loss={train_loss:.4f} steps={steps}
- gate: Gate 2 (agentharm misuse 1.0, benign >= 0.85, abstention >= 0.80)
"""


_TORCH_CU128_WHEEL = (
    "https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128"
    "-cp312-cp312-manylinux_2_28_x86_64.whl"
)
# Prebuilt qwen3_5 fused kernels, zero source builds: causal-conv1d
# cu12/torch2.10 wheel from the GitHub release assets plus fla
# (pure-python dispatch over prebuilt triton JIT kernels).
_CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0"
    "/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE"
    "-cp312-cp312-linux_x86_64.whl"
)
_PIN_FLA = "flash-linear-attention[cuda]==0.6.0"

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    # Base CUDA MUST match torch's bundled CUDA major.minor (12.8): a 12.9-system/12.8-torch
    # mix loads two cuBLAS versions in one process -> CUBLAS_STATUS_INVALID_VALUE
    # (probe 2 postmortem; cf. vLLM v0.17.0 known issue on CUDA 12.9+ mismatch).
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(_TORCH_CU128_WHEEL)
    .uv_pip_install(
        PIN_TRANSFORMERS,
        PIN_TRL,
        PIN_PEFT,
        PIN_DATASETS,
        PIN_ACCELERATE,
        PIN_BITSANDBYTES,
        PIN_HF_HUB,
        "hf_xet",
        _CAUSAL_CONV1D_WHEEL,
        _PIN_FLA,
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

data_vol = modal.Volume.from_name("dmb-flywheel", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)

app = modal.App("dmb-qwen38-safetune")


@app.local_entrypoint()
def main(mode: str = "train", dataset: str = DATASET_PATH,
         parent: str = DPO_ADAPTER_DIR, out: str = ADAPTER_DIR,
         repo: str = HF_REPO, epochs: int = EPOCHS):
    if mode not in ("probe", "train"):
        raise ValueError("mode must be 'probe' or 'train'")
    result = train.remote(probe=(mode == "probe"), dataset_path=dataset,
                          parent_dir=parent, out_dir=out, hf_repo=repo,
                          epochs=epochs)
    print(result)


@app.function(
    image=train_image,
    gpu=GPU_CHAIN,
    volumes={
        "/data": data_vol,
        "/models": models_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("dmb-hf-token")],
    timeout=21600,
)
def train(probe: bool = False, dataset_path: str = DATASET_PATH,
          parent_dir: str = DPO_ADAPTER_DIR, out_dir: str = ADAPTER_DIR,
          hf_repo: str = HF_REPO, epochs: int = EPOCHS) -> dict:
    import json
    import os
    import subprocess
    import time

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 (name is informational only)
        gpu_name = "unknown"
    gpu_class = classify_gpu(gpu_name)
    fused = detect_fused()
    print(f"gpu={gpu_name} fused={fused}", flush=True)

    if not os.path.isdir(parent_dir):
        raise RuntimeError(
            f"{parent_dir} missing: run the parent stage first")
    rows: list[dict] = []
    for line_no, line in enumerate(open(dataset_path), start=1):
        if line.strip():
            try:
                rows.append(coerce_record(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{DATASET_PATH}:{line_no}: {exc}") from exc
    print(f"pairs: {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(model, parent_dir,
                                      is_trainable=True)
    print(f"loaded parent adapter {parent_dir}; continuing on its weights",
          flush=True)

    dataset = Dataset.from_list(rows)
    full_steps = count_steps(len(rows), BATCH_SIZE, GRAD_ACCUM, epochs)
    args = SFTConfig(
        output_dir="/tmp/qwen38-safetune",
        num_train_epochs=epochs,
        max_steps=PROBE_STEPS if probe else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=10,
        save_strategy="no" if probe else "epoch",
        save_total_limit=1,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        processing_class=tokenizer,
    )
    from transformers import TrainerCallback

    abort_state: dict = {}

    class _AbortCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if (state.global_step == ABORT_CHECK_STEPS and not probe):
                check = should_abort(time.time() - train_t0,
                                     state.global_step)
                print(f"ABORT_CHECK {check}", flush=True)
                if check["abort"]:
                    abort_state["aborted"] = check
                    control.should_training_stop = True
            return control

    trainer.add_callback(_AbortCallback())
    train_t0 = time.time()
    stats = trainer.train()
    train_seconds = time.time() - train_t0
    s_per_it = train_seconds / max(1, stats.global_step)
    if abort_state.get("aborted") and not probe:
        minutes = (time.time() - t0) / 60
        print(f"ABORTED {abort_state['aborted']} after {minutes:.1f} min; no save, no push", flush=True)
        return {"mode": "aborted",
                "abort": abort_state["aborted"],
                "steps": stats.global_step,
                "minutes": round(minutes, 1),
                "gpu_name": gpu_name,
                "gpu_class": gpu_class}
    print(f"train_loss={stats.training_loss:.4f} "
          f"s_per_it={s_per_it:.2f}", flush=True)

    if probe:
        minutes = (time.time() - t0) / 60
        full = extrapolate_cost(full_steps, s_per_it, gpu_class)
        verdict = probe_verdict(s_per_it, full["cost_usd"],
                                STAGE_BUDGET_USD)
        print(f"PROBE {verdict} full_run={full}", flush=True)
        return {"mode": "probe", "pairs": len(rows),
                "steps": stats.global_step,
                "minutes": round(minutes, 1),
                "gpu_name": gpu_name, "full_run": full,
                "verdict": verdict}

    trainer.save_model(out_dir)
    models_vol.commit()

    merged = False
    try:
        print("merging adapter into bf16 base...", flush=True)
        full = trainer.model.merge_and_unload()
        full.save_pretrained(OUTPUT_DIR, max_shard_size="5GB")
        tokenizer.save_pretrained(OUTPUT_DIR)
        models_vol.commit()
        merged = True
        print(f"merged model written to {OUTPUT_DIR}", flush=True)
    except Exception as exc:  # noqa: BLE001 (OOM fallback: adapter-only)
        print(f"merge skipped ({type(exc).__name__}: {exc}); "
              f"adapter kept at {out_dir}", flush=True)

    minutes = (time.time() - t0) / 60
    commit = os.environ.get("HSEA_COMMIT", "unknown")
    card = build_model_card(stats.training_loss, stats.global_step,
                            len(rows), commit, gpu_name)
    hf_rev = None
    token = next((os.environ.get(k) for k in
                  ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
                  if os.environ.get(k)), None)
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(hf_repo, exist_ok=True, private=True)
        api.upload_folder(repo_id=hf_repo, folder_path=out_dir,
                          commit_message=f"issue-8 SafeTune adapter ({started})")
        api.upload_file(repo_id=hf_repo, path_in_repo="README.md",
                        path_or_fileobj=card.encode(),
                        commit_message="issue-8 model card")
        hf_rev = api.repo_info(hf_repo).sha
        print(f"pushed {hf_repo}@{hf_rev}", flush=True)
    else:
        print("no HF token in secret; push skipped", flush=True)

    return {
        "pairs": len(rows),
        "train_loss": stats.training_loss,
        "steps": stats.global_step,
        "minutes": round(minutes, 1),
        "s_per_it": round(s_per_it, 2),
        "adapter": out_dir,
        "merged": merged,
        "output": OUTPUT_DIR if merged else out_dir,
        "hf_repo": hf_repo if hf_rev else None,
        "hf_rev": hf_rev,
        "gpu_name": gpu_name,
        "gpu_class": gpu_class,
        "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
    }
