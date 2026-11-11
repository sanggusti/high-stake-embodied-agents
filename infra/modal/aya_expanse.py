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
  modal run infra/modal/aya_expanse.py --cmd verify
  modal run infra/modal/aya_expanse.py --cmd sft --max-steps 20   # smoke
  modal run infra/modal/aya_expanse.py --cmd sft                  # full
  modal run infra/modal/aya_expanse.py --cmd dpo --max-steps 20   # smoke
  modal run infra/modal/aya_expanse.py --cmd dpo                  # full
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint base
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint sft
  modal run infra/modal/aya_expanse.py --cmd generate --checkpoint dpo

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
PROMPTS_PATH = "/data/aya74/eval_prompts.jsonl"
OUT_ROOT = "/models/aya74"

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
    gpu="A100-80GB",
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
    gpu="A100-80GB",
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
    gpu="A10G",
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
    gpu="A10G",
    volumes={"/data": data_vol, "/models": models_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def generate(checkpoint: str = "dpo", limit: int = 0,
             include: str = "") -> dict:
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert checkpoint in ("base", "sft", "dpo"), checkpoint
    with open(PROMPTS_PATH) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    if include:
        prompts = [p for p in prompts if p["key"].startswith(include)]
    if limit > 0:
        prompts = prompts[:limit]
    print(f"generate checkpoint={checkpoint} prompts={len(prompts)}")

    tok_src = BASE_MODEL if checkpoint == "base" else f"{OUT_ROOT}/{checkpoint}"
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, revision=MODEL_REVISION if checkpoint == "base" else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda()
    if checkpoint == "sft":
        model = PeftModel.from_pretrained(model, f"{OUT_ROOT}/sft").merge_and_unload()
    elif checkpoint == "dpo":
        sft = PeftModel.from_pretrained(model, f"{OUT_ROOT}/sft").merge_and_unload()
        model = PeftModel.from_pretrained(sft, f"{OUT_ROOT}/dpo").merge_and_unload()
    model.eval()

    suffix = f"_{include.rstrip(':')}" if include else ""
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


@app.local_entrypoint()
def main(cmd: str = "verify", checkpoint: str = "dpo",
         max_steps: int = -1, epochs: int = 3, limit: int = 0,
         include: str = ""):
    if cmd == "verify":
        print(verify.remote())
    elif cmd == "sft":
        print(sft.remote(max_steps=max_steps, epochs=epochs))
    elif cmd == "dpo":
        print(dpo.remote(max_steps=max_steps))
    elif cmd == "merge":
        print(merge.remote())
    elif cmd == "generate":
        print(generate.remote(checkpoint=checkpoint, limit=limit,
                              include=include))
    else:
        raise SystemExit(f"unknown cmd {cmd!r}")
