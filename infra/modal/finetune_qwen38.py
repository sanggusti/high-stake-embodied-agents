"""AlignTune SFT: Qwen3.8-27B QLoRA on flywheel-v2 (issue #8, full loop).

Lineage: ``infra/modal/finetune.py`` (Phase D, Qwen2.5-7B) scaled to the
27B multimodal base. QLoRA (NF4 + bf16 compute) keeps the full
``Qwen3_5ForConditionalGeneration`` model on one 80GB-class GPU; LoRA
targets the language tower only (vision tower frozen, SFT rows are
text-only).

Data fixes applied at load (the pinned ``sanggusti/dmb-flywheel-v2``
files are never rewritten):
- 1,001 disasterm3 rows carry ``content`` as a list of string fragments
  (a ``list(chars)`` splat in the exporter); fragments are joined.
- OpenAI-style content blocks (``{"type": "text", "text": ...}``) are
  flattened to plain strings for the Qwen3.8 chat template.

Redo rules (binding, from docs/reports/issue-8-troubleshoot.md):
- No fused-kernel SOURCE builds. The image tries a PyPI prebuilt
  ``causal-conv1d`` wheel with ``--only-binary`` (hard failover to
  ``|| true``); when no wheel fits, training runs the reference
  PyTorch implementation with honest time math. The active path is
  logged and recorded in the return dict.
- Steps = rows / effective-batch x epochs = 8673 / 8 x 3 = 3255.
  Run ``modal run infra/modal/finetune_qwen38.py --mode probe`` FIRST:
  it trains 50 steps, measures s/it, extrapolates wall time + cost,
  and the full run only proceeds when the extrapolation is within 2x
  of the pre-run estimate. Abort and re-plan otherwise.
- GPU fallback chain on every function, never a strict single class.
- transformers/trl/torch pinned to the tested lock releases
  (requirements-train.lock), never floating main.
- Sequence 80GB-class jobs against #73/#74; never queue-war them.

Usage (needs `modal` auth + an unsandboxed shell; GPU spend applies):
  modal volume put dmb-flywheel data/flywheel-v2/sft.jsonl /flywheel-v2/sft.jsonl
  modal run infra/modal/finetune_qwen38.py --mode probe     # 50-step timing
  modal run infra/modal/finetune_qwen38.py --mode train     # full SFT
Output: adapter at /models/qwen38-flywheel-v2-lora, merged bf16 model at
/models/qwen38-flywheel-v2-merged (volume dmb-flywheel-models), plus an
HF push of the adapter repo (see HF_REPO).

Probe 1 result (2026-11-11, H200): reference impl at 8.67 s/it
(loss 1.157, healthy) — no PyPI prebuilt kernel wheel fits, and the
reference path prices SFT-3ep at ~$21-38 (NO-GO on cost). The image
now installs PREBUILT fused kernels from the GitHub release assets
(causal-conv1d 1.7.0 cu12/torch2.10) + fla 0.6.0 over prebuilt triton
JIT — zero source builds — with torch 2.10.0+cu128 (newest torch a
cu12 wheel covers). Probe 2 decides: fused (~1-2 s/it expected)
commits the full loop; reference re-measured re-runs the cost gate.

Cost estimate (probe-2 target): 3,255 steps at ~1.5 s/it = ~1.4 h +
~0.5 h setup on A100-80GB -> ~$4.75 for SFT-3ep.
"""

from __future__ import annotations

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
DATA_REPO = "sanggusti/dmb-flywheel-v2"
DATA_REVISION = "b44237e145af1f0d9cfa3f22ec36f77d22ca5393"
HF_REPO = "sanggusti/qwen38-flywheel-v2-lora"

DATASET_PATH = "/data/flywheel-v2/sft.jsonl"
ADAPTER_DIR = "/models/qwen38-flywheel-v2-lora"
OUTPUT_DIR = "/models/qwen38-flywheel-v2-merged"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048
SEED = 8

# Tested pins (requirements-train.lock, verified 2026-11-09).
# transformers 5.22.2 ships qwen3_5 modeling: no git main needed.
PIN_TRANSFORMERS = "transformers==5.22.2"
PIN_TRL = "trl==1.7.1"
PIN_TORCH = "torch==2.10.0"  # cu128: newest torch with a prebuilt cu12 causal-conv1d wheel
PIN_PEFT = "peft==0.21.2"
PIN_DATASETS = "datasets==5.1.0"
PIN_ACCELERATE = "accelerate==1.15.0"
PIN_BITSANDBYTES = "bitsandbytes==0.50.2"
PIN_HF_HUB = "huggingface-hub==1.33.0"

# GPU fallback chain (binding: never a strict single class).
GPU_CHAIN = ["A100-80GB", "H100"]

# Modal $/hr per class (from docs/reports/modal-spend.md ledger rows).
GPU_RATES = {"A100-80GB": 2.49, "H100": 3.95, "H200": 4.54}

# Pre-run planning anchor (s/it on the reference impl). The probe
# measures the truth; abort + re-plan when measured > 2x this.
ESTIMATE_S_PER_IT = 7.0
PROBE_STEPS = 50
ABORT_FACTOR = 2.0
STAGE_BUDGET_USD = 12.0

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

_ATTENTION_SUBMODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_SUBMODULES = ("gate_proj", "up_proj", "down_proj")
_VISION_HINTS = ("vision", "visual", "vit_", "vit.", "patch", "image", "mm_")
_TEXT_HINTS = ("language_model", "languagemodel", "text_model", "llm")


def normalize_content(content) -> str:
    """Flatten one message content to a plain string.

    Accepts plain strings, OpenAI-style block lists
    (``{"type": "text", "text": ...}`` or ``{"text": ...}``), and the
    disasterm3 fragment lists (bare strings, often single chars).
    Unknown block shapes raise ValueError (fail fast, never guess).
    """
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


def normalize_tool_calls(tool_calls) -> list[dict]:
    """Coerce tool calls to ``[{name, arguments: dict}]``.

    Accepts OpenAI shapes (``{function: {name, arguments}}`` or flat
    ``{name, arguments}``); arguments may be a JSON string or a mapping.
    """
    import json

    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("tool_calls must be a non-empty list")
    out: list[dict] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("tool call must be a dict")
        fn = call.get("function", call)
        if not isinstance(fn, dict):
            raise ValueError("tool call lacks a function block")
        name = fn.get("name")
        if not name:
            raise ValueError("tool call lacks a name")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except ValueError as exc:
                raise ValueError(f"tool args not JSON: {raw_args[:60]!r}") from exc
        if not isinstance(raw_args, dict):
            raise ValueError("tool args must be a JSON object")
        out.append({"name": str(name), "arguments": raw_args})
    return out


def normalize_messages(messages) -> list[dict]:
    """Coerce one chat turn list to ``[{role, content:str}]``.

    Drops empty-content turns unless they carry tool calls (assistant
    tool-call turns render via the template ``tool_calls`` path);
    rejects unknown roles and empty results.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out: list[dict] = []
    for turn in messages:
        if not isinstance(turn, dict):
            raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
        role = turn.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown role {role!r}")
        text = normalize_content(turn.get("content", ""))
        calls = normalize_tool_calls(turn["tool_calls"]) if turn.get("tool_calls") else []
        if role != "assistant" and calls:
            raise ValueError(f"tool_calls on non-assistant turn ({role})")
        if text.strip() or calls:
            item: dict = {"role": role, "content": text}
            if calls:
                item["tool_calls"] = calls
            out.append(item)
    if not out:
        raise ValueError("all turns empty after normalization")
    roles = [t["role"] for t in out]
    if "assistant" not in roles:
        raise ValueError("no assistant turn to train on")
    return out


def coerce_record(record: dict) -> dict:
    """Normalize one sft.jsonl row to ``{"messages", "source"}``."""
    if not isinstance(record, dict) or "messages" not in record:
        raise ValueError("record lacks a messages list")
    return {
        "messages": normalize_messages(record["messages"]),
        "source": str(record.get("source", "?")),
    }


def lora_regex_for(module_names: list[str]) -> str:
    """PEFT target regex covering text-tower linears only.

    Prefers modules under a text-tower path (``language_model`` etc.);
    falls back to any non-vision attention/MLP projection when the text
    path is absent (pure LM). Vision-tower linears never match.
    """
    subs = _ATTENTION_SUBMODULES + _MLP_SUBMODULES
    text = [n for n in module_names
            if any(h in n for h in _TEXT_HINTS)
            and n.rsplit(".", 1)[-1] in subs]
    pool = text or [n for n in module_names
                    if not any(h in n for h in _VISION_HINTS)
                    and n.rsplit(".", 1)[-1] in subs]
    if not pool:
        raise ValueError("no LoRA-eligible linear modules found")
    prefixes = sorted({n.rsplit(".", 1)[0] for n in pool})
    stem = "|".join(p.replace(".", r"\.") for p in prefixes)
    leaves = "|".join(subs)
    return rf"(?:{stem})\.(?:{leaves})"


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


def build_model_card(train_loss: float, steps: int, tokens: int,
                     merged: bool, commit: str, gpu: str,
                     fused: str, epochs: int = EPOCHS) -> str:
    """Adapter model card (base, data rev, config, research-only gate)."""
    return f"""---
license: other
license_name: mixed-per-source-research-only
base_model: {BASE_MODEL}
base_model_revision: {BASE_REVISION}
tags:
- lora
- disaster-response
- sft
---

# Qwen3.8-27B flywheel-v2 LoRA (issue #8)

QLoRA finetune of `{BASE_MODEL}` (@ `{BASE_REVISION}`) on the
flywheel-v2 SFT split (`{DATA_REPO}` @ `{DATA_REVISION}`,
8,673 chat pairs, {tokens:,} template tokens). Research-only: the
training mix carries CC-BY-NC-SA components (DisasterM3, HumAID,
CrisisMMD), so this adapter is gated like its data.

## Config

- method: QLoRA NF4, bf16 compute, r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}
- targets: language-tower attention + MLP projections (vision tower frozen)
- epochs={epochs} batch={BATCH_SIZE} accum={GRAD_ACCUM} lr={LEARNING_RATE} cosine
- max_seq={MAX_SEQ_LENGTH} length-grouped (no packing) seed={SEED}
- pins: {PIN_TRANSFORMERS} {PIN_TRL} {PIN_TORCH} {PIN_PEFT}
- gpu: {gpu} (fallback chain {GPU_CHAIN}); fused kernels: {fused}
- code commit: {commit}

## Result

- train_loss={train_loss:.4f} steps={steps}
- merged bf16 weights: {"pushed" if merged else "kept on the Modal volume only"}
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
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    # Tested lock pins (requirements-train.lock): transformers 5.22.2
    # ships qwen3_5 modeling, so no git-main drift. cu128 torch by
    # direct wheel URL (a bare torch pin re-resolves to PyPI's cu13.x
    # build and breaks bitsandbytes kernels).
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

app = modal.App("dmb-qwen38-finetune")


@app.local_entrypoint()
def main(mode: str = "train", epochs: int = EPOCHS):
    if mode not in ("probe", "train"):
        raise ValueError("mode must be 'probe' or 'train'")
    result = train.remote(probe=(mode == "probe"), epochs=epochs)
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
def train(probe: bool = False, epochs: int = EPOCHS) -> dict:
    import json
    import os
    import subprocess
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
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
        fused = f"fused prebuilt (causal-conv1d {cc1d}, fla {fla_v})"
    elif cc1d or fla_v:
        fused = f"PARTIAL kernels (cc1d={cc1d}, fla={fla_v})"
    else:
        fused = "reference impl (no prebuilt kernels)"
    print(f"gpu={gpu_name} fused={fused}", flush=True)

    rows: list[dict] = []
    for line_no, line in enumerate(open(DATASET_PATH), start=1):
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

    # Fail fast: every row must render under the real chat template
    # before the 27B weights load. Rows are length-sorted so unpacked
    # batches pad minimally (TRL 1.7.1 has no group_by_length).
    total_tokens = 0
    truncated = 0
    sized: list[tuple[int, dict]] = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True)
        n = len(ids["input_ids"]) if hasattr(ids, "keys") else len(ids)
        total_tokens += n
        if n > MAX_SEQ_LENGTH:
            truncated += 1
        sized.append((n, row))
    sized.sort(key=lambda pair: pair[0])
    rows = [row for _, row in sized]
    print(f"template_tokens={total_tokens} truncated>{MAX_SEQ_LENGTH}:{truncated}",
          flush=True)

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

    from torch import nn

    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    regex = lora_regex_for(linears)
    print(f"lora_regex: {regex[:160]}...", flush=True)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=regex,
    )
    dataset = Dataset.from_list(rows)
    full_steps = count_steps(len(rows), BATCH_SIZE, GRAD_ACCUM, epochs)
    args = SFTConfig(
        output_dir="/tmp/qwen38-sft",
        num_train_epochs=epochs,
        max_steps=PROBE_STEPS if probe else -1,
        # No packing: the qwen3_5 hybrid tower needs fused kernels
        # (causal-conv1d, flash-linear-attention) to keep packed
        # sequences apart; source builds are banned, so rows are
        # pre-sorted by length and padding waste stays low.
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
        packing=False,
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.model.print_trainable_parameters()
    train_t0 = time.time()
    stats = trainer.train()
    train_seconds = time.time() - train_t0
    s_per_it = train_seconds / max(1, stats.global_step)
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
            "pairs": len(rows),
            "tokens": total_tokens,
            "truncated": truncated,
            "steps": stats.global_step,
            "minutes": round(minutes, 1),
            "gpu_name": gpu_name,
            "gpu_class": gpu_class,
            "fused": fused,
            "epochs": epochs,
            "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
            "verdict": verdict,
            "full_run": full,
        }

    trainer.save_model(ADAPTER_DIR)
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
              f"adapter kept at {ADAPTER_DIR}", flush=True)

    minutes = (time.time() - t0) / 60
    commit = os.environ.get("HSEA_COMMIT", "unknown")
    card = build_model_card(stats.training_loss, stats.global_step,
                            total_tokens, merged, commit, gpu_name,
                            fused, epochs)
    hf_rev = None
    token = next((os.environ.get(k) for k in
                  ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
                  if os.environ.get(k)), None)
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(HF_REPO, exist_ok=True, private=True)
        api.upload_folder(repo_id=HF_REPO, folder_path=ADAPTER_DIR,
                          commit_message=f"issue-8 QLoRA adapter ({started})")
        api.upload_file(repo_id=HF_REPO, path_in_repo="README.md",
                        path_or_fileobj=card.encode(),
                        commit_message="issue-8 model card")
        hf_rev = api.resolve_ref(HF_REPO)
        print(f"pushed {HF_REPO}@{hf_rev}", flush=True)
    else:
        print("no HF token in secret; push skipped", flush=True)

    return {
        "pairs": len(rows),
        "tokens": total_tokens,
        "truncated": truncated,
        "epochs": epochs,
        "train_loss": stats.training_loss,
        "steps": stats.global_step,
        "minutes": round(minutes, 1),
        "s_per_it": round(s_per_it, 2),
        "merged": merged,
        "adapter": ADAPTER_DIR,
        "output": OUTPUT_DIR if merged else ADAPTER_DIR,
        "hf_repo": HF_REPO if hf_rev else None,
        "hf_rev": hf_rev,
        "gpu_name": gpu_name,
        "gpu_class": gpu_class,
        "fused": fused,
        "pins": [PIN_TRANSFORMERS, PIN_TRL, PIN_TORCH, PIN_PEFT],
    }
