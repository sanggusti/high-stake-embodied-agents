"""Modal-hosted vision model behind an OpenAI-compatible endpoint (#19, VLM #2).

Same factory shape as ``vllm_server.py`` but serving Qwen3-VL-8B-Instruct
(Apache-2.0, per docs/models.md) for the footage VLM harness. inspect-ai
reaches it as ``openai-api/modal/<model>`` with
``MODAL_BASE_URL=<serve_url>/v1``.

Verified 2026-11-09: deployed with `modal` credentials on a GPU-enabled
workspace; serves the #19 footage harness (see docs/footage-evals.md).
That footage config (A10G, no tool parser, ``--max-model-len 4096``) is
generate-only: 16384 on a 24 GB A10G leaves ~0.3 GB for KV cache and
EngineCore fails to start.

Issue #3 (mission evals): Qwen3-VL-8B is the cheap VLM workhorse, so the
mission config serves on one A100-80GB with ``--max-model-len 65536``
(weights ~16 GB BF16 + 64K KV ~10 GB — same headroom rationale as
``vllm_gemma4.py``) and Hermes tool calling:
``--enable-auto-tool-choice --tool-call-parser hermes``. NOTE this is
the OPPOSITE of Qwen3.8 (issue #1): Qwen3-VL-8B-Instruct's stock template
induces Hermes-style ``<tool_call>{"name", "arguments"}</tool_call>``
output, which ``qwen3_xml`` cannot parse — the qwen3_xml config scored
0.150 with 0 ToolEvents (calls as plain text), fixed by this one-line
redeploy. Verify parser choice with a tool_choice="auto" multi-tool
probe: a forced (named) tool_choice masks the mismatch because vLLM
constrains the output to a valid call either way.
"""

import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
APP_NAME = "dmb-vllm-qwen3vl"
MODEL_REVISION: str | None = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

# Same CUDA/vLLM pins as vllm_server.py: vLLM 0.32 fails FlashInfer JIT on
# cuda:12.4; re-pin here only after a verified vision deploy. 0.21.0 >=
# 0.20.2, so the qwen3_xml parser is available.
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
        # Required for OpenAI-style tool calling. Qwen3-VL-8B-Instruct emits
        # Hermes-style calls (see module docstring) — the OPPOSITE of
        # Qwen3.8's qwen3_xml (issue #1).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        # Mission config (issue #3): 64K working context on one A100-80GB.
        # The old footage config (A10G, 4096, no tool parser) could not run
        # multi-image ReAct episodes — see module docstring.
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
