"""AlignTune DPO: length-normalized DPO over the SFT checkpoint (issue #8).

Stage 2 of the full loop. Trains on flywheel-v2 ``dpo.jsonl`` (7,000
HumAID/CrisisMMD dispatch pairs) starting from the stage-1 SFT adapter.
Loss is TRL ``sigmoid_norm`` (length-normalized DPO: per-token average
log-ps inside the Bradley-Terry objective), the Tülu-3-vetted default
per docs/alignment-methods-posttraining.md §2.1.

Templating warning (measured 2026-11-11): ``rejected`` has only 2
distinct texts (P3-monitor for P1/P2 situations, P1-scramble for P3),
and ``chosen`` is systematically longer (median 1.75x). Two mitigations:
- length normalization is IN the loss (sigmoid_norm), not a filter;
- the IPO bake-off arm (``--mode ipo``, §2.2 insurance vs DPO overfit
  on templated pairs) trains the identical config with loss_type=ipo.

Deferred with estimate (DEVIATIONS table): the §2.1 on-policy upgrade
(generate rejected from the SFT checkpoint itself). Cost: ~7,000
situations x 1 generation on the served SFT model (~2 h A100-80GB,
~$5) + curation; skipped because the slice forbids it after SFT+DPO.

Same binding rules as stage 1 (see finetune_qwen38.py docstring):
pinned lock releases, prebuilt-wheels-only, GPU fallback chain,
probe-first with the 2x abort rule, sequence vs #73/#74.

Usage (unsandboxed shell; GPU spend applies):
  modal volume put dmb-flywheel data/flywheel-v2/dpo.jsonl /flywheel-v2/dpo.jsonl
  modal run infra/modal/dpo_qwen38.py --mode probe    # 50-step timing
  modal run infra/modal/dpo_qwen38.py --mode train    # sigmoid_norm DPO
  modal run infra/modal/dpo_qwen38.py --mode ipo      # IPO arm (if slice allows)

Cost estimate (pre-GPU): 875 steps (7,000 pairs / eff-batch 8 x 1
epoch) at ~10-16 s/it (4 forwards/step: policy+ref x chosen+rejected)
= 2.4-3.9 h + ~0.5 h setup on A100-80GB -> ~$7-11 per arm.
"""

from __future__ import annotations

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
DATA_REPO = "sanggusti/dmb-flywheel-v2"
DATA_REVISION = "b44237e145af1f0d9cfa3f22ec36f77d22ca5393"
SFT_ADAPTER_DIR = "/models/qwen38-flywheel-v2-lora"
HF_REPO = "sanggusti/qwen38-flywheel-v2-dpo"
HF_REPO_IPO = "sanggusti/qwen38-flywheel-v2-dpo-ipo"

DATASET_PATH = "/data/flywheel-v2/dpo.jsonl"
ADAPTER_DIR = "/models/qwen38-flywheel-v2-dpo"
ADAPTER_DIR_IPO = "/models/qwen38-flywheel-v2-dpo-ipo"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
BETA = 0.1
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

ESTIMATE_S_PER_IT = 13.0
PROBE_STEPS = 50
ABORT_FACTOR = 2.0
STAGE_BUDGET_USD = 11.0


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
    try:
        import tilelang  # noqa: PLC0415

        tl_v = getattr(tilelang, "__version__", "?")
    except ImportError:
        tl_v = None
    if cc1d and fla_v:
        return (f"fused prebuilt (causal-conv1d {cc1d}, fla {fla_v}, "
                f"tilelang {tl_v})")
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


def loss_curve(trainer) -> list[dict]:
    """Step/loss series from trainer log_history (JSON-safe, for plots)."""
    curve = []
    for entry in trainer.state.log_history or []:
        if "loss" not in entry:
            continue
        curve.append({"step": int(entry.get("step", 0)),
                      "loss": float(entry["loss"])})
    return curve

# UltraFeedback-ADAPT debias (survey §7.1): drop pairs whose length
# ratio is so extreme the pair teaches length, not priority. Measured
# max on flywheel-v2 is 1.92, so this drops 0 rows today while
# guarding future data revs.
LENGTH_RATIO_MAX = 3.0


def coerce_pair(record: dict) -> dict | None:
    """One dpo.jsonl row -> ``{prompt, chosen, rejected}`` chat turns.

    Returns None for rows the debias filter drops (extreme
    chosen/rejected length ratio); raises ValueError on malformed rows
    (fail fast, never guess).
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    situation = record.get("situation")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not isinstance(situation, str) or not situation.strip():
        raise ValueError("record lacks a situation string")
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("record lacks a chosen string")
    if not isinstance(rejected, str) or not rejected.strip():
        raise ValueError("record lacks a rejected string")
    ratio = len(chosen) / max(1, len(rejected))
    if ratio > LENGTH_RATIO_MAX or ratio < 1 / LENGTH_RATIO_MAX:
        return None
    prompt = [{"role": "user", "content": situation.strip()}]
    return {
        "prompt": prompt,
        "chosen": prompt + [{"role": "assistant",
                             "content": chosen.strip()}],
        "rejected": prompt + [{"role": "assistant",
                               "content": rejected.strip()}],
        "priority": str(record.get("priority", "?")),
    }


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


def build_model_card(loss_type: str, train_loss: float, steps: int,
                     pairs: int, dropped: int, commit: str,
                     gpu: str) -> str:
    """DPO adapter model card (SFT parent, data rev, research gate)."""
    return f"""---
license: other
license_name: mixed-per-source-research-only
base_model: {BASE_MODEL}
base_model_revision: {BASE_REVISION}
tags:
- lora
- disaster-response
- dpo
---

# Qwen3.8-27B flywheel-v2 DPO ({loss_type}, issue #8)

Length-normalized DPO (`{loss_type}`, beta={BETA}) over the stage-1 SFT
adapter on flywheel-v2 DPO pairs (`{DATA_REPO}` @ `{DATA_REVISION}`,
{pairs:,} pairs, {dropped} length-debias drops). Research-only: the
mix carries CC-BY-NC-SA components, gated like its data.

## Config

- parent: stage-1 SFT adapter ({SFT_ADAPTER_DIR})
- method: QLoRA NF4 continued (r={LORA_R} alpha={LORA_ALPHA}), DPO {loss_type}
- epochs={EPOCHS} batch={BATCH_SIZE} accum={GRAD_ACCUM} lr={LEARNING_RATE}
- max_seq={MAX_SEQ_LENGTH} seed={SEED}
- pins: {PIN_TRANSFORMERS} {PIN_TRL} {PIN_TORCH} {PIN_PEFT}
- gpu: {gpu} (fallback chain {GPU_CHAIN})
- code commit: {commit}

## Result

- train_loss={train_loss:.4f} steps={steps}
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
# Hopper (H100/H200) escape hatch (2026-11-12): fla 0.6.0 refuses its
# Triton gated-bwd kernel on sm_90 with Triton < 3.7.1 (wrong
# results, see fla#640) and prescribes tilelang. abi3 manylinux
# wheel, JIT-compiles in the devel image. Validated by probe.
_PIN_TILELANG = "tilelang==0.1.15"

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
        _PIN_TILELANG,
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

data_vol = modal.Volume.from_name("dmb-flywheel", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)

app = modal.App("dmb-qwen38-dpo")


@app.local_entrypoint()
def main(mode: str = "train"):
    if mode not in ("probe", "train", "ipo"):
        raise ValueError("mode must be 'probe', 'train', or 'ipo'")
    result = train.remote(
        probe=(mode == "probe"),
        loss_type="ipo" if mode == "ipo" else "sigmoid_norm",
    )
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
    timeout=43200,
)
def train(probe: bool = False, loss_type: str = "sigmoid_norm") -> dict:
    import json
    import os
    import subprocess
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer

    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    is_ipo = loss_type == "ipo"
    out_dir = ADAPTER_DIR_IPO if is_ipo else ADAPTER_DIR
    hf_repo = HF_REPO_IPO if is_ipo else HF_REPO
    try:
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 (name is informational only)
        gpu_name = "unknown"
    gpu_class = classify_gpu(gpu_name)
    fused = detect_fused()
    print(f"gpu={gpu_name} loss={loss_type} fused={fused}", flush=True)

    if not os.path.isdir(SFT_ADAPTER_DIR):
        raise RuntimeError(
            f"{SFT_ADAPTER_DIR} missing: run stage-1 SFT first")
    rows: list[dict] = []
    dropped = 0
    for line_no, line in enumerate(open(DATASET_PATH), start=1):
        if line.strip():
            try:
                pair = coerce_pair(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{DATASET_PATH}:{line_no}: {exc}") from exc
            if pair is None:
                dropped += 1
            else:
                rows.append(pair)
    print(f"pairs: {len(rows)} debias_dropped={dropped}", flush=True)

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
    # DPO continues from the SFT adapter (stage-1 output).
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR,
                                      is_trainable=True)
    print("loaded SFT adapter; continuing DPO on its weights", flush=True)

    dataset = Dataset.from_list(rows)
    full_steps = count_steps(len(rows), BATCH_SIZE, GRAD_ACCUM, EPOCHS)
    args = DPOConfig(
        output_dir="/tmp/qwen38-dpo",
        num_train_epochs=EPOCHS,
        max_steps=PROBE_STEPS if probe else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10 if probe else 25,
        save_strategy="no" if probe else "epoch",
        save_total_limit=1,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        loss_type=loss_type,
        beta=BETA,
        seed=SEED,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
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
        return {
            "mode": "probe",
            "loss_type": loss_type,
            "pairs": len(rows),
            "debias_dropped": dropped,
            "steps": stats.global_step,
            "minutes": round(minutes, 1),
            "gpu_name": gpu_name,
            "gpu_class": gpu_class,
            "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
            "verdict": verdict,
            "full_run": full,
        }

    trainer.save_model(out_dir)
    models_vol.commit()

    # No merge: merge_and_unload on bnb 4-bit linears saves packed
    # uint8 weights + quant satellites (unservable). Stages serve
    # via vLLM LoRA on the bf16 base (vllm_qwen38_ft.py).
    merged = False

    minutes = (time.time() - t0) / 60
    commit = os.environ.get("HSEA_COMMIT", "unknown")
    card = build_model_card(loss_type, stats.training_loss,
                            stats.global_step, len(rows), dropped,
                            commit, gpu_name)
    hf_rev = None
    token = next((os.environ.get(k) for k in
                  ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
                  if os.environ.get(k)), None)
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(hf_repo, exist_ok=True, private=True)
        api.upload_folder(repo_id=hf_repo, folder_path=out_dir,
                          commit_message=f"issue-8 DPO adapter ({started})")
        api.upload_file(repo_id=hf_repo, path_in_repo="README.md",
                        path_or_fileobj=card.encode(),
                        commit_message="issue-8 model card")
        hf_rev = api.repo_info(hf_repo).sha
        print(f"pushed {hf_repo}@{hf_rev}", flush=True)
    else:
        print("no HF token in secret; push skipped", flush=True)

    return {
        "loss_type": loss_type,
        "pairs": len(rows),
        "debias_dropped": dropped,
        "train_loss": stats.training_loss,
        "loss_curve": loss_curve(trainer),
        "steps": stats.global_step,
        "minutes": round(minutes, 1),
        "s_per_it": round(s_per_it, 2),
        "adapter": out_dir,
        "merged": merged,
        "output": out_dir,
        "hf_repo": hf_repo if hf_rev else None,
        "hf_rev": hf_rev,
        "gpu_name": gpu_name,
        "gpu_class": gpu_class,
        "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
    }
