"""Modal-hosted Qwen 14B (AWQ) behind an OpenAI-compatible endpoint.

Same factory as vllm_server.py: CUDA image + vLLM 0.21 + shared
HF/vLLM cache volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

Qwen2.5-14B-Instruct-AWQ (ungated): the size axis of the harder-eval
model matrix (7B vs 14B, same family). AWQ weights (~8 GB) fit the
A10G with a 16k context.
"""

import modal

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"
APP_NAME = "dmb-vllm-qwen14b"
MODEL_REVISION: str | None = None  # pin a commit for reproducibility

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
# defined inside a factory (LocalFunctionError).
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
@modal.web_server(port=8000, startup_timeout=900)
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
        "--quantization",
        "awq",
        # Required for OpenAI-style tool calling (Qwen emits Hermes format).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        # A10G fit: AWQ weights + 16k context.
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
