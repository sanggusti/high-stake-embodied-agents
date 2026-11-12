"""Tiny Aya Base full loop on Modal (issue #73): LoRA SFT + DPO + eval gen.

Base: CohereLabs/tiny-aya-base (3.35B, CC-BY-NC-4.0, research-only).
The base tokenizer ships no chat template, so this module sets the
Cohere turn format explicitly (same tokens the instruct sisters use).

Pipeline (one GPU, L4):
  1. train_sft: LoRA SFT over data/aya73/sft_train.jsonl (chat messages).
  2. train_dpo: LoRA DPO over data/aya73/dpo_train.jsonl from the SFT
     adapter (length-normalized DPO per docs/alignment-methods).
  3. push: adapters + training logs to HF (sanggusti/tinyaya73-*).
  Evals run served (infra/modal/vllm_aya73.py + inspect-ai) per the
  shared model-comparison protocol; generate() stays as an offline
  fallback only (greedy completions for every eval row).

Usage (GPU spend applies; L4 is about $0.80/hr):
  modal volume put dmb-aya73 data/aya73/sft_train.jsonl /sft_train.jsonl
  modal volume put dmb-aya73 data/aya73/dpo_train.jsonl /dpo_train.jsonl
  modal volume put dmb-aya73 data/aya73/eval_safety.jsonl /eval_safety.jsonl
  modal volume put dmb-aya73 data/aya73/eval_abstention.jsonl /eval_abstention.jsonl
  modal volume put dmb-aya73 data/aya73/eval_redteam.jsonl /eval_redteam.jsonl
  modal volume put dmb-aya73 data/aya73/eval_footage.jsonl /eval_footage.jsonl
  modal volume put dmb-aya73 data/aya73/eval_dispatch.jsonl /eval_dispatch.jsonl
  modal run infra/modal/finetune_aya73.py::smoke     # ~10 GPU-min sanity
  modal run infra/modal/finetune_aya73.py::full      # SFT + DPO + gen + push
Outputs on volume dmb-aya73: /adapters/sft, /adapters/dpo,
/completions/*.jsonl, /train_log.json.
"""

import modal

BASE_MODEL = "CohereLabs/tiny-aya-base"
BASE_REVISION = "d62ac17e241c871fc52e9a95d7b0e2615189d7d5"
HF_SFT_REPO = "sanggusti/tinyaya73-sft-lora"
HF_DPO_REPO = "sanggusti/tinyaya73-dpo-lora"

# Cohere turn format; the base tokenizer carries the same special
# tokens but no chat_template, so SFT/DPO/generation share this one.
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}<BOS_TOKEN>"
    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
    "{{ messages[0]['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}<BOS_TOKEN>{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}"
    "<|END_OF_TURN_TOKEN|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ message['content'] }}"
    "<|END_OF_TURN_TOKEN|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"
)

EVAL_FILES = (
    "eval_safety.jsonl",
    "eval_abstention.jsonl",
    "eval_redteam.jsonl",
    "eval_footage.jsonl",
    "eval_dispatch.jsonl",
)

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
    )
)

work_vol = modal.Volume.from_name("dmb-aya73", create_if_missing=True)
hf_secret = modal.Secret.from_name("hsea-hf")

app = modal.App("dmb-aya73-sft")


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _base_model():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model


@app.function(
    image=train_image,
    gpu="L4",
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def train_sft(smoke: bool = False) -> dict:
    import time
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    dataset = load_dataset("json", data_files="/work/sft_train.jsonl",
                           split="train")
    if smoke:
        dataset = dataset.select(range(min(200, len(dataset))))
    print(f"sft pairs: {len(dataset)} smoke={smoke}", flush=True)

    tokenizer = _tokenizer()
    model = _base_model()

    args = SFTConfig(
        output_dir="/tmp/aya73-sft",
        num_train_epochs=3,
        max_steps=20 if smoke else -1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        max_length=2048,
        packing=False,
        seed=0,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"sft train_loss={stats.training_loss:.4f}", flush=True)

    out = "/work/adapters/sft"
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    history = [{"step": h["step"], "loss": h["loss"]}
               for h in trainer.state.log_history if "loss" in h]
    work_vol.commit()
    minutes = (time.time() - t0) / 60
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "loss_history": history, "minutes": minutes, "output": out}


@app.function(
    image=train_image,
    gpu="L4",
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def train_dpo(smoke: bool = False) -> dict:
    import time
    from datasets import load_dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    t0 = time.time()
    raw = load_dataset("json", data_files="/work/dpo_train.jsonl",
                       split="train")
    if smoke:
        raw = raw.select(range(min(200, len(raw))))

    def to_messages(row):
        prompt_msgs = []
        if row.get("system"):
            prompt_msgs.append({"role": "system",
                                "content": row["system"]})
        prompt_msgs.append({"role": "user", "content": row["prompt"]})
        return {
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant",
                        "content": row["chosen"]}],
            "rejected": [{"role": "assistant",
                          "content": row["rejected"]}],
        }

    dataset = raw.map(to_messages,
                      remove_columns=raw.column_names)
    print(f"dpo pairs: {len(dataset)} smoke={smoke}", flush=True)

    tokenizer = _tokenizer()
    model = _base_model()
    if not smoke:
        model = PeftModel.from_pretrained(model, "/work/adapters/sft",
                                          is_trainable=True)

    args = DPOConfig(
        output_dir="/tmp/aya73-dpo",
        num_train_epochs=1,
        max_steps=20 if smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        beta=0.1,
        loss_type="sigmoid",
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        max_length=2048,
        seed=0,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None if not smoke else _lora_config(),
    )
    stats = trainer.train()
    print(f"dpo train_loss={stats.training_loss:.4f}", flush=True)

    out = "/work/adapters/dpo"
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    history = [{"step": h["step"], "loss": h["loss"]}
               for h in trainer.state.log_history if "loss" in h]
    work_vol.commit()
    minutes = (time.time() - t0) / 60
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "loss_history": history, "minutes": minutes, "output": out}


def _render_eval_prompt(tokenizer, row: dict) -> str:
    if "user_text" in row:  # footage: raw user text as-is
        messages = [{"role": "user", "content": row["user_text"]}]
    else:
        system = row.get("system")
        user = row.get("prompt", row.get("situation", ""))
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


@app.function(
    image=train_image,
    gpu="L4",
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def generate() -> dict:
    import json
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    tokenizer = _tokenizer()

    def load_adapter(path: str | None):
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if path:
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()
        model.eval()
        return model

    counts: dict[str, int] = {}
    for name, path in (("base", None), ("sft", "/work/adapters/sft"),
                       ("dpo", "/work/adapters/dpo")):
        model = load_adapter(path)
        device = next(model.parameters()).device
        for fname in EVAL_FILES:
            rows = [json.loads(line) for line in
                    open(f"/work/{fname}") if line.strip()]
            out_rows = []
            for row in rows:
                prompt = _render_eval_prompt(tokenizer, row)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=256, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)
                out_rows.append({"id": row.get("id"), "completion": text})
            dest = f"/work/completions/{name}_{fname}"
            with open(dest, "w") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r) + "\n")
            counts[f"{name}/{fname}"] = len(out_rows)
            print(f"generated {name}/{fname}: {len(out_rows)}", flush=True)
        del model
        torch.cuda.empty_cache()
    work_vol.commit()
    return {"counts": counts, "minutes": (time.time() - t0) / 60}


@app.function(
    image=train_image,
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def push_hf(train_log: dict | None = None) -> dict:
    import json
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret missing")
    api = HfApi(token=token)
    out = {}
    card = (
        "---\nlicense: cc-by-nc-4.0\nbase_model: CohereLabs/tiny-aya-base\n"
        "tags: [hsea, issue-73, lora]\n---\n"
        "HSEA issue-73 Tiny Aya Base adapter (research-only, CC-BY-NC-4.0). "
        "See the HSEA repo docs/reports/issue-73-*.md for the full loop.\n"
    )
    for repo_id, src in ((HF_SFT_REPO, "/work/adapters/sft"),
                         (HF_DPO_REPO, "/work/adapters/dpo")):
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=src,
                          commit_message="issue-73 adapter")
        with open("/tmp/README.md", "w") as fh:
            fh.write(card)
        api.upload_file(repo_id=repo_id, path_in_repo="README.md",
                        path_or_fileobj="/tmp/README.md",
                        commit_message="model card")
        out[repo_id] = "pushed"
    if train_log is not None:
        with open("/tmp/train_log.json", "w") as fh:
            json.dump(train_log, fh, indent=2)
        api.upload_file(repo_id=HF_DPO_REPO,
                        path_in_repo="train_log.json",
                        path_or_fileobj="/tmp/train_log.json",
                        commit_message="training log")
        out["train_log"] = "pushed"
    return out


@app.function(
    image=train_image,
    gpu="L4",
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=3600,
)
def merge() -> dict:
    """Merge each LoRA adapter into full bf16 weights for serving.

    vLLM cannot serve LoRA on Cohere2/commandr models (both 0.21.0 and
    the 0.34 nightly crash with VocabParallelEmbeddingWithLoRA has no
    quant_method), so evals serve merged weights instead. The DPO
    adapter already stacks on SFT (trained in place), so each adapter
    merges straight onto base.
    """
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    out = {}
    for name in ("sft", "dpo"):
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, f"/work/adapters/{name}")
        merged = model.merge_and_unload()
        dest = f"/work/merged/{name}"
        merged.save_pretrained(dest)
        _tokenizer().save_pretrained(dest)
        out[name] = dest
        print(f"merged {name} -> {dest}", flush=True)
        del model, merged
        torch.cuda.empty_cache()
    work_vol.commit()
    out["minutes"] = (time.time() - t0) / 60
    return out


@app.local_entrypoint()
def merge_only():
    print(merge.remote())


@app.local_entrypoint()
def smoke():
    """GPU sanity: 20 SFT steps + 20 DPO steps on 200-row slices."""
    sft = train_sft.remote(smoke=True)
    print("SFT smoke:", {k: v for k, v in sft.items()
                         if k != "loss_history"})
    dpo = train_dpo.remote(smoke=True)
    print("DPO smoke:", {k: v for k, v in dpo.items()
                         if k != "loss_history"})
    print(f"smoke GPU minutes: {sft['minutes'] + dpo['minutes']:.1f}")


@app.local_entrypoint()
def full():
    """Full train: SFT + DPO + HF push (evals run served, via inspect)."""
    import json

    sft = train_sft.remote(smoke=False)
    print("SFT:", {k: v for k, v in sft.items() if k != "loss_history"})
    dpo = train_dpo.remote(smoke=False)
    print("DPO:", {k: v for k, v in dpo.items() if k != "loss_history"})
    train_log = {"sft": sft, "dpo": dpo}
    import pathlib
    log_path = pathlib.Path("runs/aya73_train_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        json.dump(train_log, fh, indent=2, default=str)
    print(f"wrote {log_path}")
    res = push_hf.remote(train_log=train_log)
    print("PUSH:", res)
    total = sft["minutes"] + dpo["minutes"]
    print(f"full GPU minutes (L4): {total:.1f}")
