"""Serve the flywheel finetune (merged weights on the models volume).

Self-contained twin of vllm_server.py (Modal needs module-level serve()).
Usage: modal deploy infra/modal/vllm_flywheel.py
"""

import modal

MODEL_PATH = "/models/flywheel-v1-merged"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("dmb-vllm-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)

app = modal.App("dmb-vllm-flywheel")


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
# 20 min: 14 GB merged safetensors streams slowly off the volume.
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess

    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_PATH,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.85",
            # Skip CUDA-graph capture: volume weight streaming already makes
            # boot slow; eager inference is fast enough for 20 eval episodes.
            "--enforce-eager",
        ]
    )
