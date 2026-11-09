"""Reshard merged flywheel weights (fallback if single-file boot stalls).

Usage: modal run infra/modal/reshard.py
Output: /models/flywheel-v1-sharded/ on volume dmb-flywheel-models.
"""

import modal

SRC = "/models/flywheel-v1-merged"
DST = "/models/flywheel-v1-sharded"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("torch", "transformers", "safetensors")
)
models_vol = modal.Volume.from_name("dmb-flywheel-models", create_if_missing=True)
app = modal.App("dmb-flywheel-reshard")


@app.function(image=image, volumes={"/models": models_vol}, memory=32768, timeout=3600)
def reshard() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading", SRC, flush=True)
    model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16)
    print("saving sharded to", DST, flush=True)
    model.save_pretrained(DST, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(SRC).save_pretrained(DST)
    models_vol.commit()
    print("done", flush=True)
    return {"output": DST}
