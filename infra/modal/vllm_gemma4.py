"""Modal-hosted Gemma4-12B unified multimodal behind an OpenAI-compatible endpoint (issue #3).

Same factory shape as ``vllm_server.py`` but serving ``google/gemma-4-12B-it``
(Apache-2.0, ungated; docs/models.md shortlist pick #4) — the permissive
single-GPU multimodal baseline (encoder-free text+image+video+audio, 256K
native context, 140+ languages). inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

vLLM pin (nightly — stable 0.21.0 predates the ``gemma4_unified``
architecture; main-branch vLLM lists ``Gemma4UnifiedForConditionalGeneration``
as supported):
    vllm==0.34.0rc1.dev127+g8afb5e9ff
    wheel commit 8afb5e9ffeca2fd433f455c0e2b17814aea34c6d, built 2026-11-10
    (https://wheels.vllm.ai/nightly/ index of 2026-11-10T04:47 UTC).
Same pin as ``vllm_molmo2.py`` (issue #2): verified to contain
``vllm/model_executor/models/gemma4.py`` and the ``gemma4`` tool parser,
and to build/deploy on Modal. Re-pin to a newer nightly only after a
verified deploy.

Tool calling: Gemma 4's stock chat template (Canonical, published
2026-07-09) renders system + tools natively, and Gemma4 emits its native
``<|tool_call>call:...`` syntax, parsed server-side with
``--tool-call-parser gemma4`` (an engine parser; no custom template
needed, unlike Molmo2). Whether Gemma4 EMITS valid calls in the HSEA
ReAct loop is the issue #3 experiment — verify live, never assume.

Context: native 256K. BF16 weights (~24 GB) + 64K KV cache (~11 GB) fit
one A100-80GB comfortably; an A10G (24 GB) cannot hold the weights at
all, so this serves on A100-80GB with ``--max-model-len 65536``
(2x the proven 8B-VLM ff1 size — headroom for Gemma4's encoder-free
image tokens without paying full-256K prefill on every turn).
"""

import modal

MODEL_NAME = "google/gemma-4-12B-it"
APP_NAME = "dmb-vllm-gemma4"
MODEL_REVISION: str | None = "43c9d014b231474fa7212dbb48a762566c2b255d"

# Pinned nightly wheel (see module docstring). Direct URL so the pin is
# exact — no floating --pre resolve at image-build time. Same wheel as
# vllm_molmo2.py (issue #2).
VLLM_NIGHTLY_URL = (
    "https://wheels.vllm.ai/"
    "8afb5e9ffeca2fd433f455c0e2b17814aea34c6d/vllm-0.34.0rc1.dev127"
    "%2Bg8afb5e9ff-cp38-abi3-manylinux_2_28_x86_64.whl"
)

# Same CUDA base as vllm_server.py; only the vLLM pin differs (nightly).
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(VLLM_NIGHTLY_URL)
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
        # Native Gemma4 tool syntax (see module docstring). Stock chat
        # template renders tools; no --chat-template override needed.
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        # 64K working context on one A100-80GB (see module docstring).
        "--max-model-len",
        "65536",
        "--gpu-memory-utilization",
        "0.90",
        # Episodes accumulate up to snapshot-budget frames in one prompt.
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
