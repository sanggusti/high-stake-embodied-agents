"""AlignTune SFT: Qwen3.8-27B QLoRA on flywheel-v2 (issue #8).

Lineage: ``infra/modal/finetune.py`` (Phase D, Qwen2.5-7B) scaled to the
27B multimodal base. QLoRA (NF4 + bf16 compute) keeps the full
``Qwen3_5ForConditionalGeneration`` model on one A100-80GB; LoRA targets
the language tower only (vision tower frozen, SFT rows are text-only).

Data fixes applied at load (the pinned ``sanggusti/dmb-flywheel-v2``
files are never rewritten):
- 1,001 disasterm3 rows carry ``content`` as a list of string fragments
  (a ``list(chars)`` splat in the exporter); fragments are joined.
- OpenAI-style content blocks (``{"type": "text", "text": ...}``) are
  flattened to plain strings for the Qwen3.8 chat template.

Usage (needs `modal` auth; GPU spend applies):
  modal volume put dmb-flywheel data/flywheel-v2/sft.jsonl /flywheel-v2/sft.jsonl
  modal run infra/modal/finetune_qwen38.py
Output: adapter at /models/qwen38-flywheel-v2-lora, merged bf16 model at
/models/qwen38-flywheel-v2-merged (volume dmb-flywheel-models), plus an
HF push of the adapter repo (see HF_REPO).

Cost estimate (recorded before the first GPU op, issue #8):
  ~1,630 steps (8,673 rows, length-grouped, eff batch 8, 3 epochs),
  <=150 min on A100-80GB at ~$2.49/hr -> <=$6.25 for training.
  (Packing was abandoned: the hybrid tower needs fused kernels whose
  source builds are brittle; see the SFTConfig comment.)
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


def build_model_card(train_loss: float, steps: int, tokens: int,
                     merged: bool, commit: str) -> str:
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
- epochs={EPOCHS} batch={BATCH_SIZE} accum={GRAD_ACCUM} lr={LEARNING_RATE} cosine
- max_seq={MAX_SEQ_LENGTH} length-grouped (no packing) seed={SEED}
- code commit: {commit}

## Result

- train_loss={train_loss:.4f} steps={steps}
- merged bf16 weights: {"pushed" if merged else "kept on the Modal volume only"}
"""


_TORCH_CU129_WHEEL = (
    "https://download.pytorch.org/whl/cu129/torch-2.12.1%2Bcu129"
    "-cp312-cp312-manylinux_2_28_x86_64.whl"
)

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    # transformers + trl from main: qwen3_5 support is newer than the
    # last stable release (base config pins transformers 5.8.0.dev0).
    # cu129 torch by direct wheel URL (a bare torch pin re-resolves to
    # PyPI's cu13.x build and breaks the kernel compile against nvcc).
    .uv_pip_install(_TORCH_CU129_WHEEL)
    .uv_pip_install(
        _TORCH_CU129_WHEEL,
        "git+https://github.com/huggingface/transformers.git",
        "git+https://github.com/huggingface/trl.git",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
        "hf_xet",
        "setuptools",
        "wheel",
        "ninja",
        "packaging",
    )
    # qwen3_5 text tower mixes linear-attention layers: packed training
    # needs the fused causal conv1d kernel. Built with --no-build-
    # isolation so nvcc compiles against the installed cu129 torch
    # (isolated builds fetch cu13.x torch from PyPI and mismatch).
    .run_commands(
        "export TORCH_CUDA_ARCH_LIST='8.0' CC=gcc CXX=g++ "
        "CUDA_HOME=/usr/local/cuda; "
        "/.uv/uv pip install --python $(command -v python) "
        "--no-build-isolation causal-conv1d"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

data_vol = modal.Volume.from_name("dmb-flywheel", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)

app = modal.App("dmb-qwen38-finetune")


@app.local_entrypoint()
def main():
    result = train.remote()
    print(result)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={
        "/data": data_vol,
        "/models": models_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("dmb-hf-token")],
    timeout=10800,
)
def train() -> dict:
    import json
    import os
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
    # batches pad minimally (TRL main has no group_by_length).
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
    args = SFTConfig(
        output_dir="/tmp/qwen38-sft",
        num_train_epochs=EPOCHS,
        # No packing: the qwen3_5 hybrid tower needs fused kernels
        # (causal-conv1d, flash-linear-attention) to keep packed
        # sequences apart, and their source builds are brittle. Rows
        # are pre-sorted by length so padding waste stays low.
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=25,
        save_strategy="epoch",
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
    stats = trainer.train()
    print(f"train_loss={stats.training_loss:.4f}", flush=True)
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
                            total_tokens, merged, commit)
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
        "train_loss": stats.training_loss,
        "steps": stats.global_step,
        "minutes": round(minutes, 1),
        "merged": merged,
        "adapter": ADAPTER_DIR,
        "output": OUTPUT_DIR if merged else ADAPTER_DIR,
        "hf_repo": HF_REPO if hf_rev else None,
        "hf_rev": hf_rev,
    }
