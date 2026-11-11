"""Modal-hosted Qwen3.8-27B flywheel-v2 stage weights (issue #8).

Same factory as vllm_qwen38.py, but serves a per-stage merged bf16
checkpoint from the flywheel-models volume instead of the stock FP8
checkpoint. Shorter context than base (32K vs 262K): merged bf16
weights are ~55 GB, so the KV budget shrinks to fit one A100-80GB.

Which stage to serve comes from the ``/models/ACTIVE_CHECKPOINT``
pointer file (one line: the merged dir), written before each eval
window, e.g.::

    echo /models/qwen38-flywheel-v2-dpo-merged > /tmp/active_checkpoint
    modal volume put dmb-flywheel-models /tmp/active_checkpoint ACTIVE_CHECKPOINT
    modal deploy infra/modal/vllm_qwen38_ft.py

Requires a finished training stage (writes the merged dir + adapter).
inspect-ai reaches it as ``openai-api/modal/<name>`` with
``MODAL_BASE_URL=<serve_url>/v1``.
"""

import modal

POINTER_PATH = "/models/ACTIVE_CHECKPOINT"
DEFAULT_MODEL_PATH = "/models/qwen38-flywheel-v2-merged"
APP_NAME = "dmb-vllm-qwen38-ft"
# inspect-ai model id: openai-api/modal/qwen38-flywheel-v2.
SERVED_NAME = "qwen38-flywheel-v2"


def resolve_model_path() -> str:
    """Stage merged dir from the pointer file (default: SFT merge)."""
    try:
        with open(POINTER_PATH) as fh:
            line = fh.read().strip()
            if line:
                return line
    except OSError:
        pass
    return DEFAULT_MODEL_PATH

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    # Binding (issue #8 forensics): fallback chain, never single class.
    gpu=["A100-80GB", "H100"],
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
    import os
    import subprocess

    model_path = resolve_model_path()
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"{model_path} missing: run the training stage first, then "
            f"write its merged dir to {POINTER_PATH}")
    print(f"serving {model_path} as {SERVED_NAME}", flush=True)
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--limit-mm-per-prompt",
        '{"image": 8, "video": 2}',
    ]
    subprocess.Popen(cmd)
