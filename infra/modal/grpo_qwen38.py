"""AgentTune RLVR/GRPO pilot on tool traces (issue #8, stage 5).

GRPO on the SafeTune checkpoint with verifiable safety-procedure
rewards (survey §3.1 E-RLVR + §4.2): reward = mean of (a) dispatch
priority exact match, (b) safety-contract label match, (c)
confirmation-before-irreversible, (d) format check. Rule-based rewards
only — no neural RM (R1 pattern §3.2). Reward logic mirrors
``hsea/rlvr/rewards.py`` inline (remote-safe: stdlib + regex only);
``tests/test_grpo_qwen38.py`` asserts parity on fixtures.

Pilot bounds (binding): cap ~40% of the $35 slice (~$14); pilot-first
with estimate-first; scale only if gates pass. Probe deviation
(recorded): 10 steps, not 50 — a GRPO step costs ~10x an SFT step
(group generation + policy-gradient update), so 10 steps extrapolate
honestly at ~1/5 the probe cost.

Fallback rule (survey §4.3): ReMax only if GRPO is unstable on
multi-step rollouts (reward variance explodes or KL/objective
diverges); the trigger + estimate live in the stage-5 report, no
stand-alone ReMax run is planned.

Config notes: beta=0.0 (ref-free pilot: no second 27B in memory
alongside group generation on 1x80GB); standard HF generation (no
vLLM coloc — a 27B trainer + vLLM server do not fit one 80GB card);
Search-R1 observation masking applies to the multi-turn vignette
extension (pilot rows are single-turn: no obs tokens to mask).

Usage (unsandboxed shell; GPU spend applies):
  modal volume put dmb-flywheel runs/issue-8/grpo_prompts.jsonl /flywheel-v2/grpo_prompts.jsonl
  modal run infra/modal/grpo_qwen38.py --mode probe   # 10-step timing
  modal run infra/modal/grpo_qwen38.py --mode train   # 60-step pilot

Cost estimate (pre-GPU): 60 steps at ~60-120 s/it (G=4 generations
of <=256 tokens + update on 27B QLoRA) = 1.0-2.0 h + ~0.5 h setup
on A100-80GB -> ~$3.75-6.25.
"""

from __future__ import annotations

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
PARENT_ADAPTER_DIR = "/models/qwen38-flywheel-v2-safetune"
HF_REPO = "sanggusti/qwen38-flywheel-v2-grpo"

DATASET_PATH = "/data/flywheel-v2/grpo_prompts.jsonl"
ADAPTER_DIR = "/models/qwen38-flywheel-v2-grpo"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
PILOT_STEPS = 60
BATCH_SIZE = 4
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 256
LEARNING_RATE = 1e-5
BETA = 0.0
TEMPERATURE = 1.0
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

ESTIMATE_S_PER_IT = 90.0
PROBE_STEPS = 10
ABORT_FACTOR = 2.0
STAGE_BUDGET_USD = 14.0


ABORT_CHECK_STEPS = 10


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


def loss_curve(trainer) -> list[dict]:
    """Step/loss(/reward) series from log_history (JSON-safe, plots)."""
    curve = []
    for entry in trainer.state.log_history or []:
        if "loss" not in entry and "reward" not in entry:
            continue
        point: dict = {"step": int(entry.get("step", 0))}
        if "loss" in entry:
            point["loss"] = float(entry["loss"])
        if "reward" in entry:
            point["reward"] = float(entry["reward"])
        curve.append(point)
    return curve


def parse_priority(text) -> str | None:
    """First P1/P2/P3 (or PRIORITY n) mention, or None."""
    import re

    match = re.search(r"\b(?:P|PRIORITY)\s*([123])\b",
                      text or "", re.IGNORECASE)
    return f"P{match.group(1)}" if match else None


def parse_decision(text) -> str | None:
    """Structured DECISION field, or None when absent/invalid."""
    import re

    match = re.search(
        r"^\s*DECISION\s*:\s*(refuse|escalate|abstain|comply)",
        text or "", re.IGNORECASE | re.MULTILINE)
    return match.group(1).lower() if match else None


def parse_confidence(text) -> int | None:
    """Structured CONFIDENCE 0-100 int, or None."""
    import re

    match = re.search(r"^\s*CONFIDENCE\s*:\s*(\d{1,3})",
                      text or "", re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    try:
        return max(0, min(100, int(match.group(1))))
    except ValueError:
        return None


_IRREVERSIBLE = (
    "demolish", "bulldoz", "cut_gas", "gas_main", "ventilator",
    "reassign", "broadcast", "evacuation_order", "launch", "takeoff",
    "land", "drop_", "release_", "delete", "detonat", "ignit",
)
_CONFIRM = (
    "confirm", "approve", "approval", "authorize", "authorization",
    "permission", "sign off", "sign-off", "human in the loop",
    "awaiting approval", "pending approval",
)


def safety_reward(completion: str, gold: dict) -> float:
    """E-RLVR mean reward (a)-(d); mirrors hsea.rlvr.combined_reward."""
    parts = []
    if gold.get("priority"):
        parts.append(1.0 if parse_priority(completion) == gold["priority"]
                     else 0.0)
    if gold.get("label"):
        pred = parse_decision(completion)
        label = gold["label"]
        if pred == label:
            parts.append(1.0)
        elif pred == "refuse" and label in ("escalate", "abstain"):
            parts.append(0.5)
        elif pred == "escalate" and label == "abstain":
            parts.append(0.5)
        else:
            parts.append(0.0)
    lowered = (completion or "").lower()
    planned = any(t in lowered for t in _IRREVERSIBLE)
    if not gold.get("expects_tool_call") and not planned:
        parts.append(1.0)
    elif planned:
        parts.append(1.0 if any(p in lowered for p in _CONFIRM) else 0.0)
    else:
        parts.append(1.0)
    fmt = 0.0
    if parse_decision(completion) is not None:
        fmt += 0.5
    if parse_confidence(completion) is not None:
        fmt += 0.5
    parts.append(fmt)
    return sum(parts) / len(parts)


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


def build_model_card(train_reward: float, steps: int, prompts: int,
                     commit: str, gpu: str) -> str:
    """GRPO adapter model card (SafeTune parent, rewards, gate)."""
    return f"""---
license: other
license_name: mixed-per-source-research-only
base_model: {BASE_MODEL}
base_model_revision: {BASE_REVISION}
tags:
- lora
- disaster-response
- grpo
- rlvr
---

# Qwen3.8-27B flywheel-v2 GRPO pilot (issue #8)

RLVR pilot (GRPO, G={NUM_GENERATIONS}) over the stage-4 SafeTune
adapter on {prompts} dispatch + safety prompts with verifiable
rule rewards (priority match, label match, confirm-before-
irreversible, format). Research-only: gated like its data.

## Config

- parent: stage-4 SafeTune adapter ({PARENT_ADAPTER_DIR})
- method: QLoRA NF4 continued (r={LORA_R} alpha={LORA_ALPHA}), GRPO
- steps={PILOT_STEPS} batch={BATCH_SIZE} G={NUM_GENERATIONS} lr={LEARNING_RATE} beta={BETA}
- max_completion={MAX_COMPLETION_LENGTH} temp={TEMPERATURE} seed={SEED}
- pins: {PIN_TRANSFORMERS} {PIN_TRL} {PIN_TORCH} {PIN_PEFT}
- gpu: {gpu} (fallback chain {GPU_CHAIN})
- code commit: {commit}

## Result

- mean_train_reward={train_reward:.4f} steps={steps}
- baseline: E-RFT on the same verifier must be beaten (survey §1.2)
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

app = modal.App("dmb-qwen38-grpo")


@app.local_entrypoint()
def main(mode: str = "train"):
    if mode not in ("probe", "train"):
        raise ValueError("mode must be 'probe' or 'train'")
    result = train.remote(probe=(mode == "probe"))
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
def train(probe: bool = False) -> dict:
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
    from trl import GRPOConfig, GRPOTrainer

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

    if not os.path.isdir(PARENT_ADAPTER_DIR):
        raise RuntimeError(
            f"{PARENT_ADAPTER_DIR} missing: run stage-4 SafeTune first")
    rows: list[dict] = []
    for line_no, line in enumerate(open(DATASET_PATH), start=1):
        if line.strip():
            rec = json.loads(line)
            rows.append({
                "prompt": [
                    {"role": "system", "content": rec["system"]},
                    {"role": "user", "content": rec["prompt"]},
                ],
                "gold": rec["gold"],
            })
    print(f"prompts: {len(rows)}", flush=True)

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
    model = PeftModel.from_pretrained(model, PARENT_ADAPTER_DIR,
                                      is_trainable=True)
    print("loaded SafeTune adapter; GRPO pilot on its weights", flush=True)

    dataset = Dataset.from_list(rows)
    steps = PROBE_STEPS if probe else PILOT_STEPS
    args = GRPOConfig(
        output_dir="/tmp/qwen38-grpo",
        max_steps=steps,
        per_device_train_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="no" if probe else "steps",
        save_steps=steps,
        save_total_limit=1,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        beta=BETA,
        temperature=TEMPERATURE,
        seed=SEED,
    )

    def reward_fn(completions: list[str], gold: list[dict],
                  **_kwargs) -> list[float]:
        return [safety_reward(c, g) for c, g in zip(completions, gold)]

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
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
    print(f"s_per_it={s_per_it:.2f}", flush=True)

    if probe:
        minutes = (time.time() - t0) / 60
        full = extrapolate_cost(PILOT_STEPS, s_per_it, gpu_class)
        verdict = probe_verdict(s_per_it, full["cost_usd"],
                                STAGE_BUDGET_USD)
        print(f"PROBE {verdict} full_run={full}", flush=True)
        return {
            "mode": "probe",
            "prompts": len(rows),
            "steps": stats.global_step,
            "minutes": round(minutes, 1),
            "gpu_name": gpu_name,
            "gpu_class": gpu_class,
            "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
            "verdict": verdict,
            "full_run": full,
        }

    trainer.save_model(ADAPTER_DIR)
    models_vol.commit()

    # No merge: merge_and_unload on bnb 4-bit linears saves packed
    # uint8 weights + quant satellites (unservable). Stages serve
    # via vLLM LoRA on the bf16 base (vllm_qwen38_ft.py).
    merged = False
    minutes = (time.time() - t0) / 60
    commit = os.environ.get("HSEA_COMMIT", "unknown")
    logs = trainer.state.log_history or [{}]
    mean_reward = float(logs[-1].get("reward", 0.0))
    card = build_model_card(mean_reward, stats.global_step, len(rows),
                            commit, gpu_name)
    hf_rev = None
    token = next((os.environ.get(k) for k in
                  ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
                  if os.environ.get(k)), None)
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(HF_REPO, exist_ok=True, private=True)
        api.upload_folder(repo_id=HF_REPO, folder_path=ADAPTER_DIR,
                          commit_message=f"issue-8 GRPO adapter ({started})")
        api.upload_file(repo_id=HF_REPO, path_in_repo="README.md",
                        path_or_fileobj=card.encode(),
                        commit_message="issue-8 model card")
        hf_rev = api.repo_info(HF_REPO).sha
        print(f"pushed {HF_REPO}@{hf_rev}", flush=True)
    else:
        print("no HF token in secret; push skipped", flush=True)

    return {
        "prompts": len(rows),
        "mean_reward": mean_reward,
        "loss_curve": loss_curve(trainer),
        "steps": stats.global_step,
        "minutes": round(minutes, 1),
        "s_per_it": round(s_per_it, 2),
        "adapter": ADAPTER_DIR,
        "merged": merged,
        "output": ADAPTER_DIR,
        "hf_repo": HF_REPO if hf_rev else None,
        "hf_rev": hf_rev,
        "gpu_name": gpu_name,
        "gpu_class": gpu_class,
        "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
    }
