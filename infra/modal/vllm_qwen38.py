"""Modal-hosted Qwen3.8-27B-FP8 behind an OpenAI-compatible endpoint.

Same factory as vllm_qwen14b.py: CUDA image + pinned vLLM + shared
HF/vLLM cache volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

Qwen3.8-27B official FP8 (ungated, Apache-2.0, ~31 GB): the main Modal
workhorse + top finetune candidate (docs/models.md shortlist pick #2).
Hybrid attention (3/4 linear layers, 16 full) keeps 262K-context KV cache
at ~17 GB, so weights + full native context fit one 80GB GPU.
"""

import modal

MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
APP_NAME = "dmb-vllm-qwen38"
MODEL_REVISION: str | None = "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"

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
    gpu="A100-80GB",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/models": models_vol,
    },
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200)
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
        # Required for OpenAI-style tool calling. Qwen3.8 emits Qwen XML
        # calls (``<tool_call><function=name>...``), NOT Hermes JSON, so the
        # parser must be ``qwen3_xml`` (vLLM >= 0.20.2): ``hermes`` fails
        # silently and the calls come back as plain text (issue #1).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        # Native 262K context: hybrid attention keeps KV at ~17 GB,
        # so weights (~31 GB) + full context fit one 80GB GPU.
        "--max-model-len",
        "262144",
        "--gpu-memory-utilization",
        "0.90",
        "--limit-mm-per-prompt",
        '{"image": 8, "video": 2}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
