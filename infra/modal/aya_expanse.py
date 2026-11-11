"""Aya Expanse 8B full loop on Modal (issue #74).

Stage 1: LoRA SFT on the composed ``data/aya74/sft.jsonl`` mix (flywheel
SFT + HSEA safety/abstention contract demos). Stage 2: DPO alignment on
``data/aya74/dpo.jsonl`` over the merged SFT checkpoint. Eval
completions for every lane come from ``generate`` (base/sft/dpo).

Base model: CohereLabs/aya-expanse-8b (verified on the Hub; CC-BY-NC-4.0
+ auto-gated, so the HF secret is required). Text-only 8B: footage rows
train and eval with the ``[frame omitted]`` placeholder, matching the
flywheel convention.

Usage (Modal auth needed; GPU spend applies, estimate-first per #74):
  modal volume put dmb-flywheel data/aya74/sft.jsonl /aya74/sft.jsonl
  modal volume put dmb-flywheel data/aya74/dpo.jsonl /aya74/dpo.jsonl
  modal volume put dmb-flywheel runs/aya74/eval_prompts.jsonl /aya74/eval_prompts.jsonl
  modal volume put dmb-flywheel data/aya74/repair_sft.jsonl /aya74/repair_sft.jsonl
  modal volume put dmb-flywheel data/aya74/repair_dpo.jsonl /aya74/repair_dpo.jsonl
  modal volume put dmb-flywheel data/aya74/kto.jsonl /aya74/kto.jsonl
  modal volume put dmb-flywheel runs/aya74/eval_prompts_xstest.jsonl /aya74/eval_prompts_xstest.jsonl
  modal run infra/modal/aya_expanse.py --cmd verify
  modal run infra/modal/aya_expanse.py --cmd sft --max-steps 20   # smoke
  modal run infra/modal/aya_expanse.py --cmd sft                  # full
  modal run infra/modal/aya_expanse.py --cmd dpo --max-steps 20   # smoke
  modal run infra/modal/aya_expanse.py --cmd dpo                  # full
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint base
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint sft
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint dpo
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint dpo \\
      --prompts-path /data/aya74/eval_prompts_xstest.jsonl
  modal run infra/modal/aya_expanse.py --cmd repair --max-steps 10  # smoke
  modal run infra/modal/aya_expanse.py --cmd repair                 # Stage 4
  modal run infra/modal/aya_expanse.py --cmd kto --max-steps 10    # smoke
  modal run infra/modal/aya_expanse.py --cmd kto                    # Stage 4
  modal run infra/modal/aya_expanse.py --cmd ipo --max-steps 10    # smoke
  modal run infra/modal/aya_expanse.py --cmd ipo                    # bake-off
  modal run infra/modal/aya_expanse.py --cmd rft                    # baseline
  modal run infra/modal/aya_expanse.py --cmd grpo --max-steps 5    # smoke
  modal run infra/modal/aya_expanse.py --cmd grpo                   # Stage 5

Every function carries a GPU fallback chain (TRAIN_GPU / DUAL_GPU /
INFER_GPU); Modal tries each class in order. No function pins a
strict single class.

Outputs on volume dmb-flywheel-models: /models/aya74/{sft,dpo}/ adapters,
/models/aya74/logs/ trainer states, /models/aya74/eval_*.jsonl completions.
Every invocation is an ephemeral ``modal run`` (no deployed app to stop).
"""

import modal

BASE_MODEL = "CohereLabs/aya-expanse-8b"
MODEL_REVISION: str | None = "5062468bf9bc0c6035fd64e06274333ec127d980"
SEED = 74

SFT_PATH = "/data/aya74/sft.jsonl"
DPO_PATH = "/data/aya74/dpo.jsonl"
REPAIR_SFT_PATH = "/data/aya74/repair_sft.jsonl"
REPAIR_DPO_PATH = "/data/aya74/repair_dpo.jsonl"
KTO_PATH = "/data/aya74/kto.jsonl"
PROMPTS_PATH = "/data/aya74/eval_prompts.jsonl"
XSTEST_PROMPTS_PATH = "/data/aya74/eval_prompts_xstest.jsonl"
OUT_ROOT = "/models/aya74"

# GPU fallback chains (issue order: every Modal call carries one; Modal
# tries each class in order). TRAIN fits 40GB+ (single model + LoRA);
# DUAL needs 80GB (policy + reference); INFER fits 24GB (merged fp16).
TRAIN_GPU = ["A100-80GB", "A100-40GB", "H100", "L40S"]
DUAL_GPU = ["A100-80GB", "H100", "H200"]
INFER_GPU = ["A10G", "L40S", "A100-40GB", "A100-80GB"]

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "huggingface_hub",
        "safetune==0.1.13",
    )
)

data_vol = modal.Volume.from_name("dmb-flywheel", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hsea-hf")

app = modal.App("dmb-aya74-finetune")


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )


@app.function(
    image=train_image,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def verify() -> dict:
    """CPU preflight: data, tokenizer template, length stats, lib APIs."""
    import inspect
    import json

    import datasets
    import peft
    import torch
    import transformers
    import trl
    from transformers import AutoTokenizer

    report: dict = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
    }
    try:
        import safetune
        from safetune import evaluate, harden

        report["safetune"] = getattr(safetune, "__version__", "?")
        report["safetune_evaluate_sig"] = str(inspect.signature(evaluate.evaluate))
        report["safetune_harden_init"] = str(inspect.signature(harden.AsFTTrainer.__init__))
    except Exception as exc:  # noqa: BLE001 - preflight reports, not raises
        report["safetune_error"] = f"{type(exc).__name__}: {exc}"

    for name, path in (("sft", SFT_PATH), ("dpo", DPO_PATH)):
        ds = datasets.load_dataset("json", data_files=path, split="train")
        report[f"{name}_rows"] = len(ds)
        report[f"{name}_columns"] = ds.column_names

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION)
    assert tok.chat_template, "Aya tokenizer has no chat template"
    sample = [
        {"role": "user", "content": "Is the area mostly non flooded? [frame omitted]"},
        {"role": "assistant", "content": "no"},
    ]
    rendered = tok.apply_chat_template(sample, tokenize=False)
    report["template_sample"] = rendered[:200]

    def _tok_len(messages) -> int:
        enc = tok.apply_chat_template(messages, tokenize=True)
        if hasattr(enc, "keys") and "input_ids" in enc.keys():
            enc = enc["input_ids"]
        return len(enc)

    sft = datasets.load_dataset("json", data_files=SFT_PATH, split="train")
    lens = sorted(_tok_len(m) for m in sft["messages"])
    report["sft_len"] = {"min": lens[0], "p50": lens[len(lens) // 2],
                         "p99": lens[int(len(lens) * 0.99)], "max": lens[-1]}
    dpo = datasets.load_dataset("json", data_files=DPO_PATH, split="train")
    dlens = sorted(len(tok(p + c + r).input_ids)
                   for p, c, r in zip(dpo["prompt"], dpo["chosen"], dpo["rejected"]))
    report["dpo_len"] = {"min": dlens[0], "p50": dlens[len(dlens) // 2],
                         "p99": dlens[int(len(dlens) * 0.99)], "max": dlens[-1]}
    print(json.dumps(report, indent=2))
    return report


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def sft(max_steps: int = -1, epochs: int = 3) -> dict:
    import json
    import os

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=SFT_PATH, split="train")
    print(f"sft pairs: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Shared comparison-protocol recipe (infra/modal/finetune.py): LoRA
    # r16/a32/dropout 0.05 on all 7 projections, 3 epochs, effective
    # batch 8, LR 2e-4 cosine, max_length 4096. Seed 74 is the recorded
    # issue seed (the protocol leaves seed to the issue).
    out_dir = f"{OUT_ROOT}/sft"
    args = SFTConfig(
        output_dir="/tmp/aya74-sft",
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        bf16=True,
        max_length=4096,
        packing=False,
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"sft train_loss={stats.training_loss:.4f}")

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
    with open(f"{OUT_ROOT}/logs/sft_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json(f"{OUT_ROOT}/logs/sft_trainer_state.json")
    models_vol.commit()
    print(f"SFT adapter written to {out_dir}")
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def dpo(max_steps: int = -1) -> dict:
    import json
    import os

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files=DPO_PATH, split="train")
    print(f"dpo pairs: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        f"{OUT_ROOT}/sft", trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    # Reference = merged SFT checkpoint (frozen); policy = SFT + new LoRA.
    sft_merged = PeftModel.from_pretrained(base, f"{OUT_ROOT}/sft")
    sft_merged = sft_merged.merge_and_unload()
    sft_merged.config.use_cache = False
    sft_merged.gradient_checkpointing_enable()

    # Canonical comparison-protocol alignment pass: DPO (sigmoid,
    # beta 0.1) over the merged SFT checkpoint, 1 epoch, effective
    # batch 8, LR 5e-5 cosine. max_length 2048/1024 is the recorded
    # model-specific max length (dual-model DPO on one 80GB GPU).
    out_dir = f"{OUT_ROOT}/dpo"
    args = DPOConfig(
        output_dir="/tmp/aya74-dpo",
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        beta=0.1,
        max_length=2048,
        seed=SEED,
    )
    trainer = DPOTrainer(
        model=sft_merged,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"dpo train_loss={stats.training_loss:.4f}")

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
    with open(f"{OUT_ROOT}/logs/dpo_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json(f"{OUT_ROOT}/logs/dpo_trainer_state.json")
    models_vol.commit()
    print(f"DPO adapter written to {out_dir}")
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=INFER_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def merge() -> dict:
    """Merge SFT + DPO adapters into one servable fp dir for vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(f"{OUT_ROOT}/dpo")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16)
    sft = PeftModel.from_pretrained(model, f"{OUT_ROOT}/sft").merge_and_unload()
    merged = PeftModel.from_pretrained(sft, f"{OUT_ROOT}/dpo").merge_and_unload()
    out_dir = f"{OUT_ROOT}/merged"
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    models_vol.commit()
    print(f"merged model written to {out_dir}")
    return {"output": out_dir}


@app.function(
    image=train_image,
    gpu=INFER_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def generate(checkpoint: str = "dpo", limit: int = 0,
             include: str = "", prompts_path: str = "") -> dict:
    import json
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert checkpoint in ("base", "sft", "dpo", "repair", "kto", "ipo",
                          "grpo", "rft"), checkpoint
    src = prompts_path or PROMPTS_PATH
    with open(src) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    if include:
        prompts = [p for p in prompts if p["key"].startswith(include)]
    if limit > 0:
        prompts = prompts[:limit]
    print(f"generate checkpoint={checkpoint} prompts={len(prompts)}")

    # Adapter stacks: later stages build on earlier merges (cold-start).
    STACKS = {
        "base": [], "sft": ["sft"], "dpo": ["sft", "dpo"],
        "repair": ["sft", "dpo", "repair"], "kto": ["sft", "dpo", "kto"],
        "ipo": ["sft", "ipo"], "grpo": ["sft", "dpo", "repair", "grpo"],
        "rft": ["sft", "dpo", "rft"],
    }
    stack = STACKS[checkpoint]
    tok_src = BASE_MODEL if not stack else f"{OUT_ROOT}/{stack[-1]}"
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, revision=MODEL_REVISION if not stack else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    for adapter in stack:
        model = PeftModel.from_pretrained(
            model, f"{OUT_ROOT}/{adapter}").merge_and_unload()
    model.eval()

    if include:
        suffix = f"_{include.rstrip(':')}"
    elif os.path.basename(src) != "eval_prompts.jsonl":
        tag = os.path.basename(src).replace("eval_prompts", "").strip("_.")
        tag = tag.replace(".jsonl", "")
        suffix = f"_{tag}" if tag else ""
    else:
        suffix = ""
    out_path = f"{OUT_ROOT}/eval_{checkpoint}{suffix}.jsonl"
    n = 0
    with open(out_path, "w") as fh:
        for rec in prompts:
            messages = [{"role": "user", "content": rec["prompt"]}]
            if rec.get("system"):
                messages.insert(0, {"role": "system", "content": rec["system"]})
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                truncation=True, max_length=2048)["input_ids"].cuda()
            with torch.no_grad():
                out = model.generate(
                    inputs, do_sample=False, max_new_tokens=384,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(
                out[0][inputs.shape[1]:], skip_special_tokens=True)
            fh.write(json.dumps({"key": rec["key"], "completion": completion}) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"generated {n}/{len(prompts)}", flush=True)
    models_vol.commit()
    print(f"completions written to {out_path}")
    return {"checkpoint": checkpoint, "n": n, "output": out_path}


def _merged_base(*adapters: str):
    """Base model with adapters merged in order (cold-start stacks)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    last = adapters[-1] if adapters else None
    tok_src = BASE_MODEL if last is None else f"{OUT_ROOT}/{last}"
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, revision=MODEL_REVISION if last is None else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    for adapter in adapters:
        model = PeftModel.from_pretrained(
            model, f"{OUT_ROOT}/{adapter}").merge_and_unload()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def repair(max_steps: int = -1, epochs: int = 1) -> dict:
    """Stage 4 SafeTune repair: constrained safety-mix re-tune.

    Starts from the merged DPO checkpoint (cold-start) + fresh LoRA;
    1 epoch over ``repair_sft.jsonl`` (438 rows: unified-footer safety
    + abstention, dispatch no-footer anchors, WJ-method mix) at LR
    5e-5. Saves the ``repair`` adapter.
    """
    import json
    import os

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=REPAIR_SFT_PATH,
                           split="train")
    print(f"repair rows: {len(dataset)}")
    model, tokenizer = _merged_base("sft", "dpo")
    out_dir = f"{OUT_ROOT}/repair"
    args = SFTConfig(
        output_dir="/tmp/aya74-repair",
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=25,
        save_total_limit=1,
        bf16=True,
        max_length=2048,
        packing=False,
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, args=args,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"repair train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
    with open(f"{OUT_ROOT}/logs/repair_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json(f"{OUT_ROOT}/logs/repair_trainer_state.json")
    models_vol.commit()
    print(f"repair adapter written to {out_dir}")
    return {"rows": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def kto(max_steps: int = -1, epochs: int = 1) -> dict:
    """Stage 4 KTO arm: binary abstention labels (160 rows, 1 epoch).

    Starts from the merged DPO checkpoint + fresh LoRA. ``kto.jsonl``
    carries ``label`` True (gold demo) / False (wrong-label
    violation). Saves the ``kto`` adapter.
    """
    import json
    import os

    from datasets import load_dataset
    from trl import KTOConfig, KTOTrainer

    dataset = load_dataset("json", data_files=KTO_PATH, split="train")
    print(f"kto rows: {len(dataset)}")
    model, tokenizer = _merged_base("sft", "dpo")
    out_dir = f"{OUT_ROOT}/kto"
    args = KTOConfig(
        output_dir="/tmp/aya74-kto",
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=25,
        save_total_limit=1,
        bf16=True,
        beta=0.1,
        max_length=2048,
        seed=SEED,
    )
    trainer = KTOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"kto train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
    with open(f"{OUT_ROOT}/logs/kto_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json(f"{OUT_ROOT}/logs/kto_trainer_state.json")
    models_vol.commit()
    print(f"KTO adapter written to {out_dir}")
    return {"rows": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def ipo(max_steps: int = -1, data: str = "repair") -> dict:
    """IPO bake-off arm vs DPO overfit on templated pairs.

    Same recipe as ``dpo`` but ``loss_type="ipo"`` over the merged SFT
    checkpoint. ``data="repair"`` (default, 578 debiased pairs) is the
    slice-fitting pilot; ``data="full"`` reruns the 7,078 Phase-1
    pairs for the strict bake-off. Saves the ``ipo`` adapter.
    """
    import json
    import os

    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    assert data in ("repair", "full"), data
    path = REPAIR_DPO_PATH if data == "repair" else DPO_PATH
    dataset = load_dataset("json", data_files=path, split="train")
    print(f"ipo pairs ({data}): {len(dataset)}")
    model, tokenizer = _merged_base("sft")
    out_dir = f"{OUT_ROOT}/ipo"
    args = DPOConfig(
        output_dir="/tmp/aya74-ipo",
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        beta=0.1,
        loss_type="ipo",
        max_length=2048,
        seed=SEED,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"ipo train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
    with open(f"{OUT_ROOT}/logs/ipo_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json(f"{OUT_ROOT}/logs/ipo_trainer_state.json")
    models_vol.commit()
    print(f"IPO adapter written to {out_dir}")
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


def _gold_priority(chosen: str) -> str | None:
    """Gold P1/P2/P3 from a flywheel chosen text.

    Inline mirror of ``hsea.rlvr.rewards.priority_of``: the composed
    aya74 DPO rows carry prompt/chosen/rejected (no raw situation /
    priority keys), so RFT/GRPO read the prompt verbatim and recover
    the gold priority from the chosen head (``PRIORITY N``).
    """
    import re

    match = re.search(r"\bP\s*([123])\b|\bPRIORITY\s*([123])\b",
                      chosen or "", re.IGNORECASE)
    if not match:
        return None
    return f"P{match.group(1) or match.group(2)}"


def _dispatch_reward(completions: list[str],
                     gold: list[str]) -> list[float]:
    """Inline mirror of hsea.rlvr.rewards.rule_reward_dispatch."""
    import re

    verbs = ("task", "dispatch", "deploy", "send", "mobilize",
             "activate", "route", "assign", "launch", "scramble")
    hedges = ("notify before acting", "notify first", "before acting",
              "consider", "maybe", "perhaps", "when convenient",
              "routine cycle", "revisit", "monitor", "stand by")
    out = []
    for comp, want in zip(completions, gold):
        match = re.search(r"\bP\s*([123])\b|\bPRIORITY\s*([123])\b",
                          comp or "", re.IGNORECASE)
        pred = f"P{match.group(1) or match.group(2)}" if match else None
        total = 0.0
        if pred == want:
            total = 1.0 if any(v in comp.lower() for v in verbs) else 0.5
        upper = comp.upper()
        if "DECISION:" not in upper and "ACTION:" not in upper:
            total += 0.2
        if any(h in comp.lower() for h in hedges):
            total -= 0.3
        out.append(max(0.0, round(total, 3)))
    return out


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def grpo(max_steps: int = 100, num_generations: int = 4) -> dict:
    """Stage 5 RLVR/GRPO pilot on dispatch traces (TRL GRPO).

    Cold-starts from merged repair (falls back to merged DPO when the
    repair adapter is absent). Prompts: 200 stratified flywheel
    dispatch situations; reward: SWE-RL field-match + R1 rule shaping
    (``_dispatch_reward``). Search-R1 masking: tool-result turns ride
    in the prompt (never take loss); the pilot is single-turn so no
    tool spans appear in completions. Saves the ``grpo`` adapter.
    """
    import json
    import os
    import random

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    try:
        model, tokenizer = _merged_base("sft", "dpo", "repair")
        cold = "repair"
    except Exception:  # noqa: BLE001 - repair adapter may not exist yet
        model, tokenizer = _merged_base("sft", "dpo")
        cold = "dpo"
    print(f"grpo cold-start: {cold}")

    raw = load_dataset("json", data_files=DPO_PATH, split="train")
    buckets: dict[str, list] = {}
    for rec in raw:
        if rec.get("source") != "flywheel-v2":
            continue
        gold = _gold_priority(rec["chosen"])
        if gold is None:
            continue
        buckets.setdefault(gold, []).append(rec)
    rng = random.Random(SEED)
    sample = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        sample.extend(pool[: max(1, 200 // len(buckets))])
    rng.shuffle(sample)
    rows = [{"prompt": r["prompt"],
             "gold": _gold_priority(r["chosen"])} for r in sample[:200]]
    from datasets import Dataset
    dataset = Dataset.from_list(rows)
    print(f"grpo prompts: {len(dataset)}")

    def reward_fn(completions, gold, **kwargs) -> list[float]:
        del kwargs
        return _dispatch_reward(list(completions), list(gold))

    out_dir = f"{OUT_ROOT}/grpo"
    args = GRPOConfig(
        output_dir="/tmp/aya74-grpo",
        max_steps=max_steps,
        num_generations=num_generations,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        bf16=True,
        beta=0.04,
        max_completion_length=256,
        seed=SEED,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"grpo train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
    with open(f"{OUT_ROOT}/logs/grpo_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json(f"{OUT_ROOT}/logs/grpo_trainer_state.json")
    models_vol.commit()
    print(f"GRPO adapter written to {out_dir}")
    return {"prompts": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=INFER_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def rft(n_prompts: int = 100, n_samples: int = 4) -> dict:
    """RFT sample phase: verifier-filtered kept set (aborts under 20).

    Samples ``n_samples`` completions per dispatch prompt (temp 0.7)
    from merged DPO, keeps verifier passes (priority exact + concrete
    action + no footer), writes the kept set for ``rft_train``.
    """
    import json
    import os
    import random

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    import torch
    from datasets import Dataset, load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    raw = [r for r in load_dataset("json", data_files=DPO_PATH,
                                   split="train")
           if r.get("source") == "flywheel-v2"
           and _gold_priority(r["chosen"]) is not None]
    rng = random.Random(SEED)
    idx = rng.sample(range(len(raw)), min(n_prompts, len(raw)))
    tokenizer = AutoTokenizer.from_pretrained(f"{OUT_ROOT}/dpo")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    sft = PeftModel.from_pretrained(
        model, f"{OUT_ROOT}/sft").merge_and_unload()
    gen_model = PeftModel.from_pretrained(
        sft, f"{OUT_ROOT}/dpo").merge_and_unload()
    gen_model.eval()

    kept = []
    for i in idx:
        rec = raw[i]
        prompt = rec["prompt"]
        gold = _gold_priority(rec["chosen"])
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
            truncation=True, max_length=1024)["input_ids"].cuda()
        # Sequential samples: one sequence at a time keeps peak memory
        # flat on 24GB inference GPUs (batched num_return_sequences
        # OOMs the 8B merged model on A10G).
        for _ in range(n_samples):
            with torch.no_grad():
                out = gen_model.generate(
                    inputs, do_sample=True, temperature=0.7,
                    num_return_sequences=1, max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            comp = tokenizer.decode(out[0][inputs.shape[1]:],
                                    skip_special_tokens=True)
            if _dispatch_reward([comp], [gold])[0] >= 1.0:
                kept.append({"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": comp}]})
    print(f"rft kept {len(kept)}/{n_prompts * n_samples}")
    if len(kept) < 20:
        raise SystemExit(f"rft: only {len(kept)} kept, aborting")
    kept_path = f"{OUT_ROOT}/rft_kept.jsonl"
    with open(kept_path, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    models_vol.commit()
    print(f"rft kept set written to {kept_path}")
    return {"kept": len(kept), "output": kept_path}


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def rft_train() -> dict:
    """RFT train phase: SFT 1 epoch over the verifier-kept set.

    Runs on TRAIN_GPU: the 8B LoRA SFT phase OOMs on 24GB inference
    GPUs even at micro-batch 1 (measured twice). Reads
    ``/models/aya74/rft_kept.jsonl`` written by ``rft``.
    """
    import json

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    kept_path = f"{OUT_ROOT}/rft_kept.jsonl"
    dataset = load_dataset("json", data_files=kept_path, split="train")
    print(f"rft_train rows: {len(dataset)}")
    model, tokenizer = _merged_base("sft", "dpo")
    out_dir = f"{OUT_ROOT}/rft"
    args = SFTConfig(
        output_dir="/tmp/aya74-rft",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=25,
        save_total_limit=1,
        bf16=True,
        max_length=2048,
        packing=False,
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, args=args,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"rft train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    models_vol.commit()
    print(f"RFT adapter written to {out_dir}")
    return {"rows": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.local_entrypoint()
def main(cmd: str = "verify", checkpoint: str = "dpo",
         max_steps: int = -1, epochs: int = -1, limit: int = 0,
         include: str = "", prompts_path: str = "", data: str = "repair",
         n_prompts: int = 100, n_samples: int = 4,
         num_generations: int = 4):
    # -1 resolves to the per-command default (sft 3 epochs, repair/kto
    # 1 epoch, grpo 100 steps); explicit flags always win.
    if cmd == "verify":
        print(verify.remote())
    elif cmd == "sft":
        print(sft.remote(max_steps=max_steps,
                         epochs=epochs if epochs > 0 else 3))
    elif cmd == "dpo":
        print(dpo.remote(max_steps=max_steps))
    elif cmd == "merge":
        print(merge.remote())
    elif cmd == "generate":
        print(generate.remote(checkpoint=checkpoint, limit=limit,
                              include=include, prompts_path=prompts_path))
    elif cmd == "repair":
        print(repair.remote(max_steps=max_steps,
                            epochs=epochs if epochs > 0 else 1))
    elif cmd == "kto":
        print(kto.remote(max_steps=max_steps,
                         epochs=epochs if epochs > 0 else 1))
    elif cmd == "ipo":
        print(ipo.remote(max_steps=max_steps, data=data))
    elif cmd == "grpo":
        print(grpo.remote(max_steps=max_steps if max_steps > 0 else 100,
                          num_generations=num_generations))
    elif cmd == "rft":
        print(rft.remote(n_prompts=n_prompts, n_samples=n_samples))
        print(rft_train.remote())
    else:
        raise SystemExit(f"unknown cmd {cmd!r}")
