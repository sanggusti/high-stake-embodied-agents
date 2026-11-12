"""Tiny Aya Base stages 3-6 follow-up on Modal (issue #73).

Stage 4: SafeTune repair (constrained safety-mix re-tune) + KTO arm
(binary abstention labels). Bake-off arms: IPO (loss_type="ipo") and
ORPO (single-stage ablation from base, research §2.3 edge-lane
recipe). Stage 5: RFT verifier-filtered baseline + GRPO pilot on
dispatch traces. Generations for every checkpoint come from
``generate`` (vLLM offline, merged weights; the vLLM LoRA path
crashes on Cohere2, same as Phase 1).

Cold-start stacks (the Phase-1 DPO adapter trained IN PLACE on the
SFT adapter, so it merges straight onto base):
  base [] / sft [sft] / dpo [dpo] / repair [dpo, repair] /
  kto [dpo, kto] / ipo [sft, ipo] / orpo [orpo] /
  rft [dpo, rft] / grpo [dpo, repair, grpo]

Usage (Modal auth needed; GPU spend applies, estimate-first per #73):
  modal volume put dmb-aya73 data/aya73/repair_sft.jsonl /repair_sft.jsonl
  modal volume put dmb-aya73 data/aya73/repair_dpo.jsonl /repair_dpo.jsonl
  modal volume put dmb-aya73 data/aya73/kto.jsonl /kto.jsonl
  modal volume put dmb-aya73 runs/aya73-followup/eval_prompts.jsonl /eval_prompts.jsonl
  modal run infra/modal/finetune_aya73_followup.py --cmd verify
  modal run infra/modal/finetune_aya73_followup.py --cmd stage4   # repair+kto+ipo
  modal run infra/modal/finetune_aya73_followup.py --cmd orpo
  modal run infra/modal/finetune_aya73_followup.py --cmd rft
  modal run infra/modal/finetune_aya73_followup.py --cmd grpo --max-steps 5  # smoke
  modal run infra/modal/finetune_aya73_followup.py --cmd grpo
  modal run infra/modal/finetune_aya73_followup.py --cmd generate --checkpoints base,sft,dpo
  modal run infra/modal/finetune_aya73_followup.py --cmd push

Every function carries a GPU fallback chain (TRAIN_GPU / DUAL_GPU /
INFER_GPU, L4 cheapest-first: the 3.35B model fits 24GB everywhere,
Phase-1 DPO included). No function pins a strict single class.
Every invocation is an ephemeral ``modal run`` (no deployed app).
"""

import modal

BASE_MODEL = "CohereLabs/tiny-aya-base"
BASE_REVISION = "d62ac17e241c871fc52e9a95d7b0e2615189d7d5"
SEED = 73

REPAIR_SFT_PATH = "/work/repair_sft.jsonl"
REPAIR_DPO_PATH = "/work/repair_dpo.jsonl"
KTO_PATH = "/work/kto.jsonl"
DPO_PATH = "/work/dpo_train.jsonl"
PROMPTS_PATH = "/work/eval_prompts.jsonl"
OUT_ADAPTERS = "/work/adapters"
OUT_COMPLETIONS = "/work/completions"

HF_REPOS = {
    "repair": "sanggusti/tinyaya73-repair-lora",
    "kto": "sanggusti/tinyaya73-kto-lora",
    "ipo": "sanggusti/tinyaya73-ipo-lora",
    "orpo": "sanggusti/tinyaya73-orpo-lora",
    "rft": "sanggusti/tinyaya73-rft-lora",
    "grpo": "sanggusti/tinyaya73-grpo-lora",
}

STACKS = {
    "base": [], "sft": ["sft"], "dpo": ["dpo"],
    "repair": ["dpo", "repair"], "kto": ["dpo", "kto"],
    "ipo": ["sft", "ipo"], "orpo": ["orpo"],
    "rft": ["dpo", "rft"], "grpo": ["dpo", "repair", "grpo"],
}

# GPU fallback chains (issue order: every Modal call carries one; Modal
# tries each class in order; L4 first: 3.35B fits 24GB on every leg).
TRAIN_GPU = ["L4", "A10G", "A100-40GB", "A100-80GB"]
DUAL_GPU = ["L4", "A10G", "A100-40GB", "A100-80GB"]
INFER_GPU = ["L4", "A10G", "A100-40GB"]

# Cohere turn format; byte-parity with finetune_aya73.CHAT_TEMPLATE
# (pinned by tests/test_aya73_followup_modal.py).
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

# vLLM nightly pin shared with vllm_aya73.py (verified 2026-11-10 on
# the merged Phase-1 weights).
VLLM_NIGHTLY_URL = (
    "https://wheels.vllm.ai/"
    "8afb5e9ffeca2fd433f455c0e2b17814aea34c6d/vllm-0.34.0rc1.dev127"
    "%2Bg8afb5e9ff-cp38-abi3-manylinux_2_28_x86_64.whl"
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

gen_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        VLLM_NIGHTLY_URL,
        "transformers",
        "datasets",
        "peft",
        "huggingface_hub",
    )
)

work_vol = modal.Volume.from_name("dmb-aya73", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache",
                                      create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache",
                                        create_if_missing=True)
hf_secret = modal.Secret.from_name("hsea-hf")

app = modal.App("dmb-aya73-followup")


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


def _merged_base(*adapters: str):
    """Base model with adapters merged in order (cold-start stacks)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    last = adapters[-1] if adapters else None
    if last is None:
        tokenizer = _tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{OUT_ADAPTERS}/{last}")
        tokenizer.chat_template = CHAT_TEMPLATE
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    for adapter in adapters:
        model = PeftModel.from_pretrained(
            model, f"{OUT_ADAPTERS}/{adapter}").merge_and_unload()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer


def _to_messages(row):
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


@app.function(
    image=train_image,
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def verify() -> dict:
    """CPU preflight: data, template, trainer APIs, dispatch schema."""
    import inspect
    import json

    import datasets
    import peft
    import torch
    import transformers
    import trl

    report: dict = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
    }
    from trl import GRPOConfig, GRPOTrainer, KTOConfig, KTOTrainer

    available = [n for n in dir(trl) if "Trainer" in n or "Config" in n]
    report["trl_trainers"] = sorted(available)
    # TRL 1.17 ships no ORPO: the orpo leg uses the custom HF Trainer
    # subclass in this module (loss math in _orpo_loss, unit-tested).
    report["orpo_impl"] = "custom Trainer (no TRL ORPO)"

    report["trainers"] = sorted(
        c.__name__ for c in (KTOConfig, KTOTrainer,
                             GRPOConfig, GRPOTrainer))
    assert "max_prompt_length" not in inspect.signature(
        GRPOConfig).parameters, "GRPOConfig still takes max_prompt_length"

    for name, path in (("repair_sft", REPAIR_SFT_PATH),
                       ("repair_dpo", REPAIR_DPO_PATH),
                       ("kto", KTO_PATH), ("dpo", DPO_PATH)):
        ds = datasets.load_dataset("json", data_files=path, split="train")
        report[f"{name}_rows"] = len(ds)
        report[f"{name}_columns"] = ds.column_names
    kto = datasets.load_dataset("json", data_files=KTO_PATH, split="train")
    assert set(kto["label"]) <= {True, False}, "kto labels must be bool"
    assert sum(kto["label"]) > 0 and sum(kto["label"]) < len(kto)

    fly = [r for r in datasets.load_dataset(
        "json", data_files=DPO_PATH, split="train")
        if (r.get("provenance") or {}).get("src") == "flywheel-v2"]
    missing = [r for r in fly
               if (r.get("provenance") or {}).get("priority")
               not in ("P1", "P2", "P3")]
    assert fly, "no flywheel rows in dpo_train"
    assert not missing, f"{len(missing)} flywheel rows lack priority"
    report["flywheel_rows"] = len(fly)

    tok = _tokenizer()
    assert tok.chat_template, "explicit Cohere template missing"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False, add_generation_prompt=True)
    assert rendered.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
    report["template_ok"] = True
    print(json.dumps(report, indent=2))
    return report


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def repair(max_steps: int = -1, epochs: int = 1) -> dict:
    """Stage 4 SafeTune repair: constrained safety-mix re-tune.

    Cold-starts from merged DPO + fresh LoRA; 1 epoch over
    ``repair_sft.jsonl`` (334 rows: gate-clean safety + single-verdict
    abstention cal, dispatch anchors, WJ-method mix) at LR 5e-5.
    """
    import json
    import os

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=REPAIR_SFT_PATH,
                           split="train")
    print(f"repair rows: {len(dataset)}")
    model, tokenizer = _merged_base("dpo")
    out_dir = f"{OUT_ADAPTERS}/repair"
    args = SFTConfig(
        output_dir="/tmp/aya73-repair",
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
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"repair train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs("/work/logs", exist_ok=True)
    with open("/work/logs/repair_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json("/work/logs/repair_trainer_state.json")
    work_vol.commit()
    print(f"repair adapter written to {out_dir}")
    return {"rows": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def kto(max_steps: int = -1, epochs: int = 1) -> dict:
    """Stage 4 KTO arm: binary abstention labels, 1 epoch.

    Cold-starts from merged DPO + fresh LoRA. ``kto.jsonl`` carries
    ``label`` True (cal gold demo) / False (wrong-label violation or
    mined DPO verdict-drift completion).
    """
    import json
    import os

    from datasets import load_dataset
    from trl import KTOConfig, KTOTrainer

    dataset = load_dataset("json", data_files=KTO_PATH, split="train")
    print(f"kto rows: {len(dataset)}")
    model, tokenizer = _merged_base("dpo")
    out_dir = f"{OUT_ADAPTERS}/kto"
    args = KTOConfig(
        output_dir="/tmp/aya73-kto",
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
    os.makedirs("/work/logs", exist_ok=True)
    with open("/work/logs/kto_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json("/work/logs/kto_trainer_state.json")
    work_vol.commit()
    print(f"KTO adapter written to {out_dir}")
    return {"rows": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def ipo(max_steps: int = -1, data: str = "repair") -> dict:
    """IPO bake-off arm vs DPO overfit on templated pairs.

    Same recipe as Phase-1 DPO but ``loss_type="ipo"`` over merged
    SFT. ``data="repair"`` (default, 526 debiased pairs) is the
    slice-fitting pilot; ``data="full"`` reruns the 7,054 Phase-1
    pairs for the strict bake-off.
    """
    import json
    import os

    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    assert data in ("repair", "full"), data
    path = REPAIR_DPO_PATH if data == "repair" else DPO_PATH
    raw = load_dataset("json", data_files=path, split="train")
    dataset = raw.map(_to_messages, remove_columns=raw.column_names)
    print(f"ipo pairs ({data}): {len(dataset)}")
    model, tokenizer = _merged_base("sft")
    out_dir = f"{OUT_ADAPTERS}/ipo"
    args = DPOConfig(
        output_dir="/tmp/aya73-ipo",
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
    os.makedirs("/work/logs", exist_ok=True)
    with open("/work/logs/ipo_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json("/work/logs/ipo_trainer_state.json")
    work_vol.commit()
    print(f"IPO adapter written to {out_dir}")
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


def _mean_response_logprob(logits, labels) -> "torch.Tensor":
    """Mean logprob over response tokens (labels != -100), per row."""
    import torch
    import torch.nn.functional as F

    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs.gather(-1, labels.clamp(min=0).unsqueeze(-1))
    token_lp = token_lp.squeeze(-1)
    mask = labels.ne(-100).to(token_lp.dtype)
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def _log_odds(mean_lp) -> "torch.Tensor":
    """log(P/(1-P)) from mean logprob P (stable for mean_lp <= 0)."""
    import torch

    return mean_lp - torch.log1p(-mean_lp.exp().clamp(max=1 - 1e-7))


def _orpo_loss(mean_w, mean_l, sft_loss, beta: float = 0.1):
    """ORPO objective: SFT loss + beta * odds-ratio loss (Hong et al.).

    L_OR = -log σ(log(odds_w / odds_l)); total = L_SFT + β·L_OR.
    Returns (total, or_loss, accuracy).
    """
    import torch
    import torch.nn.functional as F

    log_ratio = _log_odds(mean_w) - _log_odds(mean_l)
    or_loss = -F.logsigmoid(log_ratio).mean()
    acc = (mean_w > mean_l).float().mean()
    return sft_loss + beta * or_loss, or_loss, acc


def _chat_ids(tokenizer, messages, **kwargs) -> list:
    """apply_chat_template ids across shapes (list vs BatchEncoding).

    transformers 5 returns a BatchEncoding (not a dict subclass, but
    with .keys()/.get()); older versions return a plain id list.
    """
    out = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(out, "keys") and "input_ids" in (out.keys() or []):
        ids = out["input_ids"]
        return [int(x) for x in ids]
    return [int(x) for x in out]


def _orpo_tokenize(tokenizer, prompt_msgs, completion_msgs,
                   max_length: int = 2048) -> dict:
    """Tokenize prompt+completion; labels mask the prompt (-100)."""
    prompt_ids = _chat_ids(tokenizer, prompt_msgs,
                           add_generation_prompt=True)
    full_ids = _chat_ids(tokenizer, prompt_msgs + completion_msgs,
                         add_generation_prompt=False)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "labels": labels}


def _orpo_map_row(tokenizer, row, max_length: int = 2048) -> dict:
    """One preference pair -> flat chosen/rejected id lists."""
    mapped = _to_messages(row)
    chosen = _orpo_tokenize(tokenizer, mapped["prompt"],
                            mapped["chosen"], max_length)
    rejected = _orpo_tokenize(tokenizer, mapped["prompt"],
                              mapped["rejected"], max_length)
    return {"chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"]}


def _orpo_collate(tokenizer, rows: list[dict]) -> dict:
    """Pad chosen/rejected sides separately; boolean masks."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def _pad(key, value):
        return pad_sequence(
            [torch.tensor(r[key]) for r in rows], batch_first=True,
            padding_value=value)
    ci = _pad("chosen_input_ids", tokenizer.pad_token_id)
    ri = _pad("rejected_input_ids", tokenizer.pad_token_id)
    return {"chosen_input_ids": ci,
            "chosen_attention_mask": ci.ne(tokenizer.pad_token_id),
            "chosen_labels": _pad("chosen_labels", -100),
            "rejected_input_ids": ri,
            "rejected_attention_mask": ri.ne(tokenizer.pad_token_id),
            "rejected_labels": _pad("rejected_labels", -100)}


class _ORPOTrainerMixin:
    """compute_loss for the ORPO objective (mixed into HF Trainer)."""

    def orpo_compute_loss(self, model, inputs, beta: float = 0.1):
        out_w = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"])
        out_l = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"])
        mean_w = _mean_response_logprob(out_w.logits,
                                        inputs["chosen_labels"])
        mean_l = _mean_response_logprob(out_l.logits,
                                        inputs["rejected_labels"])
        return _orpo_loss(mean_w, mean_l, out_w.loss, beta=beta)


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def orpo(max_steps: int = -1) -> dict:
    """ORPO single-stage ablation arm (research §2.3 edge-lane recipe).

    Trains from BASE (not SFT) + fresh LoRA over the full 7,054-pair
    Phase-1 DPO mix in one stage: SFT loss + odds-ratio preference
    loss, beta 0.1, 1 epoch. Compare against the SFT+DPO stack.

    TRL 1.17 dropped ORPOTrainer, so this leg uses a small HF Trainer
    subclass (two forwards per step: chosen for SFT + odds, rejected
    for odds). Loss math mirrors ``_orpo_loss`` (unit-tested).
    """
    import json
    import os

    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    raw = load_dataset("json", data_files=DPO_PATH, split="train")
    print(f"orpo pairs: {len(raw)}")
    base, tokenizer = _merged_base()
    model = get_peft_model(base, _lora_config())
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = raw.map(lambda row: _orpo_map_row(tokenizer, row),
                      remove_columns=raw.column_names)
    # Arrow stores the id lists as plain sequences; drop the index
    # column and materialize lists so the collator sees rows of ints.
    dataset = dataset.flatten_indices()

    class ORPOTrainer(_ORPOTrainerMixin, Trainer):
        def compute_loss(self, model, inputs, **kwargs):
            del kwargs
            total, or_loss, acc = self.orpo_compute_loss(model, inputs)
            if self.state.global_step % 10 == 0:
                print(f"step={self.state.global_step} "
                      f"total={total.item():.4f} "
                      f"or={or_loss.item():.4f} acc={acc.item():.3f}",
                      flush=True)
            return total

    out_dir = f"{OUT_ADAPTERS}/orpo"
    args = TrainingArguments(
        output_dir="/tmp/aya73-orpo",
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
        seed=SEED,
        # Custom chosen_*/rejected_* columns feed compute_loss via the
        # collator; Trainer's default column pruning would drop them.
        remove_unused_columns=False,
    )
    trainer = ORPOTrainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=lambda rows: _orpo_collate(tokenizer, rows),
        processing_class=tokenizer)
    stats = trainer.train()
    print(f"orpo train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs("/work/logs", exist_ok=True)
    with open("/work/logs/orpo_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json("/work/logs/orpo_trainer_state.json")
    work_vol.commit()
    print(f"ORPO adapter written to {out_dir}")
    return {"pairs": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


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


def _flywheel_dispatch(n: int, seed: int = SEED) -> list[dict]:
    """Stratified dispatch (situation, gold priority) from dpo_train."""
    import random
    from collections import Counter

    from datasets import load_dataset

    raw = [r for r in load_dataset("json", data_files=DPO_PATH,
                                   split="train")
           if (r.get("provenance") or {}).get("src") == "flywheel-v2"]
    buckets: dict[str, list] = {}
    for rec in raw:
        buckets.setdefault(rec["provenance"]["priority"], []).append(rec)
    rng = random.Random(seed)
    total = len(raw)
    quotas = {k: max(1, round(n * len(v) / total))
              for k, v in buckets.items()}
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=quotas.get)  # type: ignore[arg-type]
        quotas[biggest] -= 1
    sample = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        sample.extend(pool[:quotas[key]])
    rng.shuffle(sample)
    print(f"dispatch sample: {dict(Counter(r['provenance']['priority'] for r in sample))}")
    return [{"prompt": r["prompt"],
             "gold": r["provenance"]["priority"]} for r in sample[:n]]


@app.function(
    image=train_image,
    gpu=INFER_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def rft(n_prompts: int = 100, n_samples: int = 4,
        min_kept: int = 20) -> dict:
    """RFT sample phase: verifier-filtered kept set (aborts under 20).

    Samples ``n_samples`` completions per dispatch prompt (temp 0.7)
    from merged DPO, keeps verifier passes (priority exact + concrete
    action + no footer), writes the kept set for ``rft_train``.
    Smoke path lowers ``min_kept`` (plumbing check, not a kept set).
    """
    import json
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    sample = _flywheel_dispatch(n_prompts)
    tokenizer = _tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    gen_model = PeftModel.from_pretrained(
        model, f"{OUT_ADAPTERS}/dpo").merge_and_unload()
    gen_model.eval()

    kept = []
    for row in sample:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, return_tensors="pt",
            truncation=True, max_length=1024)["input_ids"].cuda()
        for _ in range(n_samples):
            with torch.no_grad():
                out = gen_model.generate(
                    inputs, do_sample=True, temperature=0.7,
                    num_return_sequences=1, max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            comp = tokenizer.decode(out[0][inputs.shape[1]:],
                                    skip_special_tokens=True)
            if _dispatch_reward([comp], [row["gold"]])[0] >= 1.0:
                kept.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": comp}]})
    print(f"rft kept {len(kept)}/{n_prompts * n_samples}")
    if len(kept) < min_kept:
        raise SystemExit(f"rft: only {len(kept)} kept, aborting")
    kept_path = "/work/rft_kept.jsonl"
    with open(kept_path, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    work_vol.commit()
    print(f"rft kept set written to {kept_path}")
    return {"kept": len(kept), "output": kept_path}


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=7200,
)
def rft_train(max_steps: int = -1) -> dict:
    """RFT train phase: SFT 1 epoch over the verifier-kept set.

    The 3.35B LoRA SFT phase fits 24GB (the #74 A100 split was
    8B-specific). Cold-starts from merged DPO + fresh LoRA.
    """
    import json
    import os

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    kept_path = "/work/rft_kept.jsonl"
    dataset = load_dataset("json", data_files=kept_path, split="train")
    print(f"rft_train rows: {len(dataset)}")
    model, tokenizer = _merged_base("dpo")
    out_dir = f"{OUT_ADAPTERS}/rft"
    args = SFTConfig(
        output_dir="/tmp/aya73-rft",
        num_train_epochs=1,
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
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    stats = trainer.train()
    print(f"rft train_loss={stats.training_loss:.4f}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs("/work/logs", exist_ok=True)
    with open("/work/logs/rft_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json("/work/logs/rft_trainer_state.json")
    work_vol.commit()
    print(f"RFT adapter written to {out_dir}")
    return {"rows": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=train_image,
    gpu=DUAL_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def grpo(max_steps: int = 100, num_generations: int = 4) -> dict:
    """Stage 5 RLVR/GRPO pilot on dispatch traces (TRL GRPO).

    Cold-starts from merged repair (falls back to merged DPO when the
    repair adapter is absent). Prompts: 198 stratified flywheel
    dispatch situations; reward: SWE-RL field-match + R1 rule shaping
    (``_dispatch_reward``). Search-R1 masking: tool-result turns ride
    in the prompt (never take loss); the pilot is single-turn so no
    tool spans appear in completions. Saves the ``grpo`` adapter.
    """
    import json
    import os

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    try:
        model, tokenizer = _merged_base("dpo", "repair")
        cold = "repair"
    except Exception:  # noqa: BLE001 - repair adapter may not exist yet
        model, tokenizer = _merged_base("dpo")
        cold = "dpo"
    print(f"grpo cold-start: {cold}")

    dataset = Dataset.from_list(_flywheel_dispatch(198))
    print(f"grpo prompts: {len(dataset)}")

    def reward_fn(completions, gold, **kwargs) -> list[float]:
        del kwargs
        return _dispatch_reward(list(completions), list(gold))

    out_dir = f"{OUT_ADAPTERS}/grpo"
    args = GRPOConfig(
        output_dir="/tmp/aya73-grpo",
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
    os.makedirs("/work/logs", exist_ok=True)
    with open("/work/logs/grpo_log_history.json", "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    trainer.state.save_to_json("/work/logs/grpo_trainer_state.json")
    work_vol.commit()
    print(f"GRPO adapter written to {out_dir}")
    return {"prompts": len(dataset), "train_loss": stats.training_loss,
            "output": out_dir}


@app.function(
    image=gen_image,
    gpu=INFER_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol,
              "/root/.cache/vllm": vllm_cache_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def generate(checkpoints: str = "base,sft,dpo", limit: int = 0) -> dict:
    """Batched greedy generations via vLLM offline (merged weights).

    Same engine + greedy sampler + chat template as the Phase-1
    served evals; the only transport change is offline batching
    (no HTTPS server). One invocation serves several checkpoints
    sequentially; outputs land in /work/completions/eval_<ckpt>.jsonl.
    """
    import gc
    import json
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    wanted = [c.strip() for c in checkpoints.split(",") if c.strip()]
    assert wanted and set(wanted) <= set(STACKS), wanted
    with open(PROMPTS_PATH) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    if limit > 0:
        prompts = prompts[:limit]
    print(f"generate checkpoints={wanted} prompts={len(prompts)}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def merged_dir(ckpt: str) -> tuple[str, str]:
        stack = STACKS[ckpt]
        if ckpt in ("sft", "dpo") and not stack[1:]:
            reuse = f"/work/merged/{ckpt}"
            if os.path.isdir(reuse):
                return reuse, reuse
        if not stack:
            return BASE_MODEL, BASE_MODEL
        dest = f"/tmp/merged-{ckpt}"
        if not os.path.isdir(dest):
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, revision=BASE_REVISION,
                torch_dtype=torch.bfloat16)
            for adapter in stack:
                model = PeftModel.from_pretrained(
                    model, f"{OUT_ADAPTERS}/{adapter}").merge_and_unload()
            model.save_pretrained(dest)
            tok.save_pretrained(dest)
            del model
            torch.cuda.empty_cache()
            print(f"merged {ckpt} -> {dest}", flush=True)
        return dest, dest

    results: dict[str, dict] = {}
    os.makedirs(OUT_COMPLETIONS, exist_ok=True)
    for ckpt in wanted:
        model_path, tok_path = merged_dir(ckpt)
        revision = BASE_REVISION if model_path == BASE_MODEL else None
        llm = LLM(model=model_path, revision=revision,
                  tokenizer=tok_path if tok_path != BASE_MODEL else None,
                  chat_template=CHAT_TEMPLATE,
                  gpu_memory_utilization=0.90, max_model_len=8192,
                  seed=SEED)
        conversations = []
        for rec in prompts:
            messages = []
            if rec.get("system"):
                messages.append({"role": "system",
                                 "content": rec["system"]})
            messages.append({"role": "user", "content": rec["prompt"]})
            conversations.append(messages)
        params = SamplingParams(temperature=0.0, seed=SEED,
                                max_tokens=max(
                                    r.get("max_tokens", 256)
                                    for r in prompts))
        outputs = llm.chat(conversations, params, use_tqdm=True)
        out_path = f"{OUT_COMPLETIONS}/eval_{ckpt}.jsonl"
        with open(out_path, "w") as fh:
            for rec, out in zip(prompts, outputs):
                fh.write(json.dumps({
                    "key": rec["key"],
                    "completion": out.outputs[0].text}) + "\n")
        results[ckpt] = {"n": len(prompts), "output": out_path}
        print(f"wrote {out_path}", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    work_vol.commit()
    return results


@app.function(
    image=train_image,
    gpu=INFER_GPU,
    volumes={"/work": work_vol,
              "/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=10800,
)
def generate_hf(checkpoint: str = "dpo", limit: int = 0) -> dict:
    """Transformers sequential fallback when vLLM offline is unavailable.

    Same prompts, greedy sampler, merged weights; slower. One
    checkpoint per call.
    """
    import json
    import os

    import torch

    assert checkpoint in STACKS, checkpoint
    with open(PROMPTS_PATH) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    if limit > 0:
        prompts = prompts[:limit]
    print(f"generate_hf checkpoint={checkpoint} prompts={len(prompts)}")
    stack = STACKS[checkpoint]
    model, tokenizer = _merged_base(*stack)
    model = model.cuda().eval()
    os.makedirs(OUT_COMPLETIONS, exist_ok=True)
    out_path = f"{OUT_COMPLETIONS}/eval_{checkpoint}.jsonl"
    n = 0
    with open(out_path, "w") as fh:
        for rec in prompts:
            messages = []
            if rec.get("system"):
                messages.append({"role": "system",
                                 "content": rec["system"]})
            messages.append({"role": "user", "content": rec["prompt"]})
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", truncation=True,
                max_length=2048)["input_ids"].cuda()
            with torch.no_grad():
                out = model.generate(
                    inputs, do_sample=False,
                    max_new_tokens=rec.get("max_tokens", 256),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(
                out[0][inputs.shape[1]:], skip_special_tokens=True)
            fh.write(json.dumps({"key": rec["key"],
                                 "completion": completion}) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"generated {n}/{len(prompts)}", flush=True)
    work_vol.commit()
    print(f"completions written to {out_path}")
    return {"checkpoint": checkpoint, "n": n, "output": out_path}


@app.function(
    image=train_image,
    volumes={"/work": work_vol},
    secrets=[hf_secret],
    timeout=3600,
)
def push_followup(train_log: dict | None = None) -> dict:
    """Push the six follow-up adapters + training logs to HF."""
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
        "HSEA issue-73 Tiny Aya Base follow-up adapter (research-only, "
        "CC-BY-NC-4.0). See the HSEA repo docs/reports/issue-73-*.md "
        "for the full loop.\n"
    )
    for name, repo_id in HF_REPOS.items():
        src = f"{OUT_ADAPTERS}/{name}"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=src,
                          commit_message=f"issue-73 {name} adapter")
        with open("/tmp/README.md", "w") as fh:
            fh.write(card)
        api.upload_file(repo_id=repo_id, path_in_repo="README.md",
                        path_or_fileobj="/tmp/README.md",
                        commit_message="model card")
        out[repo_id] = "pushed"
    if train_log is not None:
        with open("/tmp/followup_train_log.json", "w") as fh:
            json.dump(train_log, fh, indent=2)
        api.upload_file(repo_id=HF_REPOS["repair"],
                        path_in_repo="followup_train_log.json",
                        path_or_fileobj="/tmp/followup_train_log.json",
                        commit_message="follow-up training log")
        out["train_log"] = "pushed"
    return out


@app.local_entrypoint()
def main(cmd: str = "verify", checkpoints: str = "base,sft,dpo",
         max_steps: int = -1, epochs: int = -1, limit: int = 0,
         data: str = "repair", n_prompts: int = 100, n_samples: int = 4,
         min_kept: int = 20, num_generations: int = 4):
    # -1 resolves to the per-command default (repair/kto 1 epoch,
    # grpo 100 steps); explicit flags always win.
    if cmd == "verify":
        print(verify.remote())
    elif cmd == "repair":
        print(repair.remote(max_steps=max_steps,
                            epochs=epochs if epochs > 0 else 1))
    elif cmd == "kto":
        print(kto.remote(max_steps=max_steps,
                         epochs=epochs if epochs > 0 else 1))
    elif cmd == "ipo":
        print(ipo.remote(max_steps=max_steps, data=data))
    elif cmd == "orpo":
        print(orpo.remote(max_steps=max_steps))
    elif cmd == "rft":
        print(rft.remote(n_prompts=n_prompts, n_samples=n_samples,
                         min_kept=min_kept))
        print(rft_train.remote(max_steps=max_steps))
    elif cmd == "rft-train":
        print(rft_train.remote(max_steps=max_steps))
    elif cmd == "grpo":
        print(grpo.remote(max_steps=max_steps if max_steps > 0 else 100,
                          num_generations=num_generations))
    elif cmd == "stage4":
        print(repair.remote(max_steps=max_steps,
                            epochs=epochs if epochs > 0 else 1))
        print(kto.remote(max_steps=max_steps,
                         epochs=epochs if epochs > 0 else 1))
        print(ipo.remote(max_steps=max_steps, data=data))
    elif cmd == "generate":
        print(generate.remote(checkpoints=checkpoints, limit=limit))
    elif cmd == "generate_hf":
        for ckpt in [c.strip() for c in checkpoints.split(",")]:
            print(generate_hf.remote(checkpoint=ckpt, limit=limit))
    elif cmd == "push":
        print(push_followup.remote())
    else:
        raise SystemExit(f"unknown cmd {cmd!r}")
