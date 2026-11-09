"""Modal-hosted vision model behind an OpenAI-compatible endpoint (#19, VLM #2).

Same factory shape as ``vllm_server.py`` but serving Qwen3-VL-8B-Instruct
(Apache-2.0, per docs/models.md) for the footage VLM harness. Footage tasks
are generate-only, so no ``--tool-call-parser`` is needed. inspect-ai reaches
it as ``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

UNTESTED: needs `modal` credentials (`modal setup`) and a GPU-enabled
workspace. See docs/inference.md.
"""

import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
APP_NAME = "dmb-vllm-qwen3vl"
MODEL_REVISION: str | None = None  # pin a commit for reproducibility

# Same CUDA/vLLM pins as vllm_server.py: vLLM 0.32 fails FlashInfer JIT on
# cuda:12.4; re-pin here only after a verified vision deploy.
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
        # Vision encoder is VRAM-hungry; footage prompts carry one image.
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--limit-mm-per-prompt",
        '{"image": 2}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
