"""Modal-hosted custom model behind an OpenAI-compatible endpoint.

Pattern follows Modal's vLLM OpenAI-compatible serving example: a CUDA
container image with vLLM installed, Hugging Face + vLLM caches on Modal
Volumes, and `vllm serve` exposed over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

UNTESTED: needs `modal` credentials (`modal setup`) and a GPU-enabled
workspace. See docs/inference.md.
"""

import modal

# Default teacher. VL attempt 2026-10-28: Qwen2.5-VL-7B-Instruct on A10G
# (16k ctx, mm-limit 8) serves but will not emit Hermes tool calls
# (outputs pseudo-calls like arm_drone_motors() as text; forced
# tool_choice works). Flywheel v1 is text-policy: the LM navigates on
# edge REPORTS, DMB_VISION=text. Serve a finetune via
# infra/modal/vllm_flywheel.py (same factory, merged-weights path).
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
APP_NAME = "dmb-vllm-openai-compatible"
MODEL_REVISION: str | None = None  # pin a commit for reproducibility

# Image pins follow Modal's verified vLLM OpenAI-compatible example: newer
# vLLM (0.32) fails FlashInfer JIT (ninja error) on cuda:12.4, crashing
# EngineCore before port 8000 binds.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)

# NOTE: keep serve() at module level — Modal cannot deploy functions
# defined inside a factory (LocalFunctionError). The finetune twin
# (vllm_flywheel.py) duplicates this block with a different model path.
app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/models": models_vol,
    },
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        # Required for OpenAI-style tool calling (Qwen emits Hermes format).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        # A10G fit: encoder + graphs OOM at 32k; episodes are short.
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        # Episodes accumulate up to snapshot-budget frames in one prompt.
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
