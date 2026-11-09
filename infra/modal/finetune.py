"""Flywheel LoRA finetune on Modal (Phase D).

Trains Qwen2.5-7B-Instruct on success-filtered teacher pairs
(chat ``messages`` format from ``dmb.data.export``), merges the adapter,
and writes a full servable model dir to the flywheel volume.

Usage (needs `modal` auth; GPU spend applies):
  modal volume put dmb-flywheel data/flywheel-v1/sft.jsonl /flywheel-v1/sft.jsonl
  modal run infra/modal/finetune.py
Output: /models/flywheel-v1-merged/ on volume dmb-flywheel-models.
"""

import modal

DATASET_PATH = "/data/flywheel-v1/sft.jsonl"
OUTPUT_DIR = "/models/flywheel-v1-merged"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

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
    )
)

data_vol = modal.Volume.from_name("dmb-flywheel", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)

app = modal.App("dmb-flywheel-finetune")


@app.function(
    image=train_image,
    gpu="A10G",
    volumes={"/data": data_vol, "/models": models_vol},
    timeout=3600,
)
def train() -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"pairs: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = SFTConfig(
        output_dir="/tmp/dmb-sft",
        num_train_epochs=3,
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
        # No dataset_text_field: dataset carries chat "messages", which
        # SFTTrainer formats with the tokenizer chat template.
        seed=0,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
    )
    stats = trainer.train()
    print(f"train_loss={stats.training_loss:.4f}")

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    models_vol.commit()
    print(f"merged model written to {OUTPUT_DIR}")
    return {"pairs": len(dataset), "train_loss": stats.training_loss, "output": OUTPUT_DIR}
