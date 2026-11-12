"""Modal-hosted Tiny Aya Base issue-73 checkpoints (OpenAI-compatible).

Serves ONE checkpoint per deploy (env AYA73_CKPT in base/sft/dpo):
vLLM cannot serve LoRA on Cohere2/commandr models (both 0.21.0 and the
0.34 nightly crash in compute_logits with VocabParallelEmbeddingWithLoRA
has no quant_method), so the SFT/DPO checkpoints serve as merged full
weights from the training volume (/work/merged/*, see merge_only in
infra/modal/finetune_aya73.py). Base serves straight from HF.

Deploy all three in parallel (one L4 each)::

  AYA73_CKPT=base modal deploy infra/modal/vllm_aya73.py
  AYA73_CKPT=sft modal deploy infra/modal/vllm_aya73.py
  AYA73_CKPT=dpo modal deploy infra/modal/vllm_aya73.py

inspect-ai reaches each as openai-api/modal/tinyaya73-<ckpt> with
MODAL_BASE_URL=<serve_url>/v1.

LICENSE: CC-BY-NC-4.0 + Cohere Labs AUP, research-only. Tool flags are
set (vLLM rejects tool_choice="auto" without them); the template
renders no tools, so mission sims measure the honest no-call text
floor (HSEA_VISION=text), same as the #4 Tiny Aya lane.
"""

import os

import modal

MODEL_NAME = "CohereLabs/tiny-aya-base"
MODEL_REVISION = "d62ac17e241c871fc52e9a95d7b0e2615189d7d5"
CKPT = os.environ.get("AYA73_CKPT", "base")
assert CKPT in ("base", "sft", "dpo"), f"bad AYA73_CKPT={CKPT!r}"
APP_NAME = f"dmb-vllm-aya73-{CKPT}"
SERVED_NAME = f"tinyaya73-{CKPT}"

# Same Cohere turn format the trainers use (also persisted as
# chat_template.jinja in the merged dirs; the flag pins it either way).
CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ messages[0]['content'] }}<|END_OF_TURN_TOKEN|>{% set loop_messages = messages[1:] %}{% else %}<BOS_TOKEN>{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>{% elif message['role'] == 'assistant' %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"""  # noqa: E501

# Nightly pin shared with vllm_molmo2.py / vllm_gemma4.py (verified
# 2026-11-10).
VLLM_NIGHTLY_URL = (
    "https://wheels.vllm.ai/"
    "8afb5e9ffeca2fd433f455c0e2b17814aea34c6d/vllm-0.34.0rc1.dev127"
    "%2Bg8afb5e9ff-cp38-abi3-manylinux_2_28_x86_64.whl"
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(VLLM_NIGHTLY_URL, "huggingface_hub")
    # Bake the checkpoint choice into the image: deploy-time env vars do
    # not propagate to containers, and the remote re-import must resolve
    # the same CKPT (else every server would serve base weights).
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "AYA73_CKPT": CKPT})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
work_vol = modal.Volume.from_name("dmb-aya73", create_if_missing=True)
hf_secret = modal.Secret.from_name("hsea-hf")

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu="L4",
    # Keep one container warm while deployed: inspect-ai clients cannot
    # survive the cold-start redirect dance, and evals stall on a scaled
    # to zero server. Stop the app right after the eval battery.
    min_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/work": work_vol,
    },
    secrets=[hf_secret],
    timeout=10800,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess

    with open("/tmp/aya73_chat_template.jinja", "w") as fh:
        fh.write(CHAT_TEMPLATE)

    if CKPT == "base":
        model_arg = [MODEL_NAME, "--revision", MODEL_REVISION]
    else:
        model_arg = [f"/work/merged/{CKPT}"]

    cmd = [
        "vllm", "serve", *model_arg,
        "--host", "0.0.0.0", "--port", "8000",
        "--chat-template", "/tmp/aya73_chat_template.jinja",
        "--served-model-name", SERVED_NAME,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
    ]
    subprocess.Popen(cmd)
