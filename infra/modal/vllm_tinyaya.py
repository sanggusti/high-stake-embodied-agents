"""Modal-hosted Tiny Aya L2 Thinker behind an OpenAI-compatible endpoint.

Same factory as vllm_qwen14b.py: CUDA image + pinned vLLM + shared
HF/vLLM cache volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

CohereLabs Tiny Aya L2 Thinker (3.35B, 32K ctx): small Cohere pick for
the edge lane (docs/models.md shortlist pick #5; issue #4). LICENSE:
**CC-BY-NC-4.0 + Cohere Labs AUP — research-only, NOT for product**
(see docs/inference.md). The repo is HF-gated (click-through license),
so the function carries the ``hsea-hf`` Modal secret (``HF_TOKEN``) for
the weight download. The chat template has no tool support, but the tool
flags are still set (vLLM rejects tool_choice="auto" without them);
tools are silently dropped and forest_fire scores are the no-call text
floor (run with ``HSEA_VISION=text``).
"""

import modal

MODEL_NAME = "CohereLabs/tiny-aya-l2-thinker"
APP_NAME = "dmb-vllm-tinyaya"
MODEL_REVISION: str | None = "a692a7bff493f7a0f2e0c821b4a795234c31d10e"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)

# Gated repo (CC-BY-NC click-through): the container needs HF_TOKEN to
# download weights. Create once via:
#   modal secret create hsea-hf HF_TOKEN=<user-access-token>
hf_secret = modal.Secret.from_name("hsea-hf")

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
    secrets=[hf_secret],
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess

    # Tool flags are set even though the Tiny Aya chat template renders
    # no tools: without them vLLM REJECTS inspect's tool_choice="auto"
    # requests (400) instead of running the episode. With them, the
    # model gets a plain-text prompt (tools silently dropped by the
    # template) and emits no parseable calls, so mission evals measure
    # the honest no-call text floor (HSEA_VISION=text). Parser choice
    # is arbitrary (nothing to parse); hermes is the generic default.
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
        # Card-stated 32K context (input + output).
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
