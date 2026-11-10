"""Modal-hosted Aya Expanse 8B base + issue-74 finetune (vLLM).

Two serverless endpoints behind OpenAI-compatible URLs so inspect-ai
reaches them as ``openai-api/modal/<name>`` with ``MODAL_BASE_URL``.
``serve_base`` loads the gated CohereLabs repo (HF secret required);
``serve_merged`` loads the merged SFT+DPO dir from the models volume.
Serverless: endpoints cold-start on first request and scale to zero.

Usage:
  modal deploy infra/modal/vllm_aya74.py   # note both serve URLs
  MODAL_BASE_URL=<merged-url>/v1 inspect eval evals/safety.py --model openai-api/modal/aya74-merged
  modal app stop dmb-vllm-aya74            # 0 tasks after each eval block
"""

import modal

BASE_MODEL = "CohereLabs/aya-expanse-8b"
MODEL_REVISION: str | None = "5062468bf9bc0c6035fd64e06274333ec127d980"
MERGED_PATH = "/models/aya74/merged"
APP_NAME = "dmb-vllm-aya74"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("hsea-hf")

app = modal.App(APP_NAME)


def _serve_cmd(model: str, revision: str | None) -> list[str]:
    # Aya Expanse 8B: 8K native context, text-only. No tool parser: the
    # #74 inspect lanes (safety, abstention, redteam) are plain
    # system+user generate() calls with no tool_choice.
    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
    ]
    if revision:
        cmd += ["--revision", revision]
    return cmd


@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret],
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200)
def serve_base():
    import subprocess

    subprocess.Popen(_serve_cmd(BASE_MODEL, MODEL_REVISION))


@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/models": models_vol,
    },
    secrets=[hf_secret],
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200)
def serve_merged():
    import subprocess

    subprocess.Popen(_serve_cmd(MERGED_PATH, None))
