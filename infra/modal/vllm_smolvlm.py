"""Modal-hosted SmolVLM-256M-Instruct behind an OpenAI-compatible endpoint.

Same factory as vllm_qwen14b.py: CUDA image + pinned vLLM + shared
HF/vLLM cache volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

SmolVLM-256M-Instruct (ungated, Apache-2.0, idefics3 arch): extreme-edge
VLM floor (docs/models.md shortlist pick #6; issue #4). 8K context
(``text_config.max_position_embeddings`` 8192). The chat template has
no tool support, but the tool flags are still set (vLLM rejects
tool_choice="auto" without them); tools are silently dropped and
forest_fire scores are the no-call vision floor. Fits anywhere (card:
one image inference under 1 GB GPU RAM); L4 for uniformity with the
rest of the edge lane.
"""

import modal

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
APP_NAME = "dmb-vllm-smolvlm"
MODEL_REVISION: str | None = "7e3e67edbbed1bf9888184d9df282b700a323964"

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
    gpu="L4",
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

    # Tool flags are set even though the SmolVLM chat template renders
    # no tools: without them vLLM REJECTS inspect's tool_choice="auto"
    # requests (400) instead of running the episode. With them, the
    # model gets prompt text (+ images) with tools silently dropped by
    # the template and emits no parseable calls, so mission evals
    # measure the honest no-call vision floor (HSEA_VISION default).
    # Parser choice is arbitrary (nothing to parse); hermes is the
    # generic default.
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        # Native 8K context.
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
