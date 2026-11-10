"""Modal-hosted SmolLM3-3B behind an OpenAI-compatible endpoint.

Same factory as vllm_qwen14b.py: CUDA image + pinned vLLM + shared
HF/vLLM cache volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

SmolLM3-3B (ungated, Apache-2.0): small text baseline for the edge lane
(docs/models.md; issue #4). 64K trained context (card claims 128K via
YARN extrapolation); GQA keeps 64K KV small enough for one L4. The
repo's ``chat_template.jinja`` renders Hermes-style XML tool calls
(``<tool_call>{"name", "arguments"}</tool_call>``), so the parser is
``hermes``. Text-only: run mission evals with ``HSEA_VISION=text``.
"""

import modal

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
APP_NAME = "dmb-vllm-smollm3"
MODEL_REVISION: str | None = "a07cc9a04f16550a088caea529712d1d335b0ac1"

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

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        # Required for OpenAI-style tool calling. SmolLM3's chat template
        # renders Hermes XML calls, so the parser must be ``hermes``.
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        # 64K trained context (card: up to 128K via YARN).
        "--max-model-len",
        "65536",
        "--gpu-memory-utilization",
        "0.90",
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
