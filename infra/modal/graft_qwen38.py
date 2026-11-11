"""Graft finetuned text weights onto the base multimodal shell (issue #8).

Training loads the TEXT-only tower (``Qwen3_5ForCausalLM``: all SFT
rows are text, vision frozen), so merged checkpoints carry
``Qwen3_5TextConfig`` — which vLLM 0.21's multimodal qwen3_5 runner
rejects. The graft restores servability WITHOUT changing text
behavior: every text tensor comes from the merged stage dir, every
vision tensor bit-identical from the base repo (frozen during
training), config.json verbatim from base. Text math is unchanged
by construction (same tower, same forward).

CPU-only (streams safetensors shard-by-shard, ~10 GB peak), ~$0.20.

Usage (unsandboxed shell):
  modal run infra/modal/graft_qwen38.py --merged qwen38-flywheel-v2-dpo-merged
Output: /models/<merged>-grafted (servable by vllm_qwen38_ft.py).
"""

from __future__ import annotations

import modal

BASE_REPO = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
SHARD_BYTES = 5 * 1024**3

image = (
    modal.Image.from_registry("python:3.12-slim")
    .uv_pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .uv_pip_install("safetensors==0.7.0", "huggingface-hub==1.33.0")
)

models_vol = modal.Volume.from_name("dmb-flywheel-models")
hf_cache_vol = modal.Volume.from_name("dmb-huggingface-cache")

app = modal.App("dmb-qwen38-graft")


def resolve_tensor(merged_names: set[str], base_name: str) -> str | None:
    """Merged tensor name for a base tensor (exact or prefix-stripped).

    Text-only merges store ``model.layers...``; the multimodal base
    stores ``language_model.model.layers...``. Vision tensors never
    resolve (correctly: they come from base).
    """
    if base_name in merged_names:
        return base_name
    prefix = "language_model."
    if base_name.startswith(prefix) and base_name[len(prefix):] in merged_names:
        return base_name[len(prefix):]
    return None


@app.local_entrypoint()
def main(merged: str = "qwen38-flywheel-v2-dpo-merged"):
    print(graft.remote(merged))


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={
        "/models": models_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=7200,
)
def graft(merged: str) -> dict:
    import glob
    import json
    import os

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    merged_dir = f"/models/{merged}"
    out_dir = f"{merged_dir}-grafted"
    os.makedirs(out_dir, exist_ok=True)

    # Base repo files from the shared HF cache (training warmed it).
    base_dirs = glob.glob(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/"
        f"snapshots/{BASE_REVISION}")
    if not base_dirs:
        raise RuntimeError("base snapshot missing from HF cache volume")
    base_dir = base_dirs[0]
    base_index = json.load(open(f"{base_dir}/model.safetensors.index.json"))
    weight_map: dict[str, str] = base_index["weight_map"]

    merged_shards = sorted(glob.glob(f"{merged_dir}/*.safetensors"))
    if not merged_shards:
        raise RuntimeError(f"no safetensors in {merged_dir}")
    merged_handles = [safe_open(p, framework="pt", device="cpu") for p in merged_shards]
    merged_names: set[str] = set()
    for h in merged_handles:
        merged_names.update(h.keys())
    base_handles: dict[str, object] = {}

    def base_slice(name: str):
        shard = weight_map[name]
        if shard not in base_handles:
            base_handles[shard] = safe_open(f"{base_dir}/{shard}",
                                            framework="pt", device="cpu")
        return base_handles[shard].get_slice(name)

    def merged_slice(name: str):
        for h in merged_handles:
            if name in h.keys():
                return h.get_slice(name)
        raise KeyError(name)

    # Graft shard-by-shard (peak RAM ~ one shard + slices).
    out_index: dict[str, str] = {}
    shard_id, shard_bytes = 1, 0
    pending: dict[str, object] = {}
    n_merged, n_base, vision_from_base = 0, 0, 0

    def flush():
        nonlocal shard_id, shard_bytes, pending
        if not pending:
            return
        path = f"{out_dir}/model-{shard_id:05d}-of-00005.safetensors"
        save_file(pending, path)
        for k in pending:
            out_index[k] = os.path.basename(path)
        shard_id += 1
        shard_bytes = 0
        pending = {}

    for name in weight_map:
        src = resolve_tensor(merged_names, name)
        tensor = (merged_slice(src)[:] if src else base_slice(name)[:])
        if src:
            n_merged += 1
        else:
            n_base += 1
            if name.split(".")[0] in ("vision", "visual", "vit", "image",
                                      "multimodal"):
                vision_from_base += 1
        nbytes = tensor.nelement() * tensor.element_size()
        if shard_bytes + nbytes > SHARD_BYTES and pending:
            flush()
        pending[name] = tensor
        out_index[name] = ""  # set properly at flush
        shard_bytes += nbytes
    flush()

    total = 0
    for p in glob.glob(f"{out_dir}/*.safetensors"):
        with safe_open(p, framework="pt", device="cpu") as h:
            for k in h.keys():
                t = h.get_slice(k)[:]
                total += t.nelement() * t.element_size()
    n_shards = shard_id - 1
    # Rename shards to the true count (index must match file names).
    final_index: dict[str, str] = {}
    files = sorted(glob.glob(f"{out_dir}/*.safetensors"))
    for i, path in enumerate(files, start=1):
        want = f"{out_dir}/model-{i:05d}-of-{n_shards:05d}.safetensors"
        if path != want:
            os.rename(path, want)
        for name, old in out_index.items():
            if old == os.path.basename(path):
                final_index[name] = os.path.basename(want)
    json.dump({"metadata": {"total_size": total},
               "weight_map": final_index},
              open(f"{out_dir}/model.safetensors.index.json", "w"),
              indent=2)

    # Config verbatim from base (multimodal shell) + tokenizer from merged.
    for name in ("config.json", "generation_config.json"):
        src_p, dst_p = f"{base_dir}/{name}", f"{out_dir}/{name}"
        if os.path.exists(src_p):
            with open(src_p, "rb") as fh_in, open(dst_p, "wb") as fh_out:
                fh_out.write(fh_in.read())
    for name in os.listdir(merged_dir):
        if name.startswith("tokenizer") or name.endswith(".jinja"):
            with open(f"{merged_dir}/{name}", "rb") as fh_in, \
                    open(f"{out_dir}/{name}", "wb") as fh_out:
                fh_out.write(fh_in.read())
    models_vol.commit()

    return {
        "merged_dir": merged_dir,
        "out_dir": out_dir,
        "tensors_merged": n_merged,
        "tensors_base": n_base,
        "vision_from_base": vision_from_base,
        "shards": n_shards,
        "total_bytes": total,
    }
