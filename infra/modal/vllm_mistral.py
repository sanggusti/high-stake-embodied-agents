"""Modal-hosted Mistral 7B behind an OpenAI-compatible endpoint.

Same factory as vllm_server.py (Qwen): CUDA image + vLLM 0.21 + shared
HF/vLLM cache volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

Mistral-7B-Instruct-v0.3 (Apache 2.0, ungated): second text family for
the harder-eval model matrix. Tool calling via the mistral parser.
"""

import modal

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
APP_NAME = "dmb-vllm-mistral"
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
        # Required for OpenAI-style tool calling (Mistral format).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "mistral",
        # A10G fit: episodes are short.
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
