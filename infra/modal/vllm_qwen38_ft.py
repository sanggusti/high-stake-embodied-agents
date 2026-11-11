"""Modal-hosted Qwen3.8-27B stage adapters via vLLM LoRA (issue #8).

Serves the bf16 BASE (Qwen/Qwen3.8-27B, 55 GB, from the shared HF
cache) with one stage adapter applied at runtime (vLLM ``--enable-lora``).
No weight merging: ``merge_and_unload`` on bitsandbytes 4-bit linears
saves packed uint8 weights + quant satellites (proven by the DPO
inspect: 2,480 satellite tensors, fp32 norms), which no server can
load as bf16. Each stage adapter is CUMULATIVE (DPO continued the SFT
adapter in place), so one ``--lora-modules`` entry serves the full
stage. Text behavior is exactly the trained model.

Which adapter comes from the ``/models/ACTIVE_ADAPTER`` pointer file
(one line: the adapter dir), written before each eval window, e.g.::

    echo /models/qwen38-flywheel-v2-dpo > /tmp/active_adapter
    modal volume put dmb-flywheel-models /tmp/active_adapter ACTIVE_ADAPTER
    modal deploy infra/modal/vllm_qwen38_ft.py

inspect-ai reaches it as ``openai-api/modal/qwen38-flywheel-v2``
(the LoRA name) with ``MODAL_BASE_URL=<serve_url>/v1``.
"""

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
POINTER_PATH = "/models/ACTIVE_ADAPTER"
DEFAULT_ADAPTER = "/models/qwen38-flywheel-v2-dpo"
APP_NAME = "dmb-vllm-qwen38-ft"
# LoRA name (what inspect-ai requests) + base rename (unused).
SERVED_NAME = "qwen38-flywheel-v2"
BASE_SERVED_NAME = "qwen38-base-bf16"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.22.2")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)

app = modal.App(APP_NAME)


def resolve_adapter() -> str:
    """Stage adapter dir from the pointer file (default: DPO)."""
    try:
        with open(POINTER_PATH) as fh:
            line = fh.read().strip()
            if line:
                return line
    except OSError:
        pass
    return DEFAULT_ADAPTER


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

    adapter = resolve_adapter()
    if not os.path.isdir(adapter):
        raise RuntimeError(
            f"{adapter} missing: run the training stage first, then "
            f"write its adapter dir to {POINTER_PATH}")
    print(f"serving base {BASE_MODEL} + LoRA {adapter} as {SERVED_NAME}",
          flush=True)
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--revision",
        BASE_REVISION,
        "--served-model-name",
        BASE_SERVED_NAME,
        "--tokenizer",
        adapter,
        "--trust-remote-code",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--max-loras",
        "1",
        "--lora-modules",
        f"{SERVED_NAME}={adapter}",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    subprocess.run(cmd, check=True)
