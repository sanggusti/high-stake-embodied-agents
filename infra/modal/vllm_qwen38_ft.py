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

Cold-start forensics (2026-11-11): the first deploy's task 1 crashed
in BASE weight load (fused-qkv shape assert) while task 2, identical
code/weights 25 min later, loaded fine from the warm HF cache
(52.09 GiB in 39.8 s) — a transient cold-cache materialization race,
not a config error. ``serve()`` therefore pre-flights the snapshot
(``snapshot_download`` + safetensors inventory gate) before exec'ing
vLLM, so the server never starts on a half-materialized cache.
Eval-critical flags match the working base serve (``vllm_qwen38.py``):
qwen3_xml tool parser + mm limits; only ``--max-model-len`` differs
(32768 vs 262144 — bf16 weights leave no room for 262K KV).
"""

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
POINTER_PATH = "/models/ACTIVE_ADAPTER"
DEFAULT_ADAPTER = "/models/qwen38-flywheel-v2-dpo"
APP_NAME = "dmb-vllm-qwen38-lora"
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
    import json
    import os
    import subprocess

    adapter = resolve_adapter()
    if not os.path.isdir(adapter):
        raise RuntimeError(
            f"{adapter} missing: run the training stage first, then "
            f"write its adapter dir to {POINTER_PATH}")
    # Cold-start gate (2026-11-11 forensics): fully materialize +
    # inventory the pinned snapshot BEFORE vLLM starts, so a retry
    # never reads a half-downloaded cache (task-1 shape assert).
    # NOTE (tunnel forensics): this function MUST Popen-and-return
    # like vllm_qwen38.py — the blocking subprocess.run revision
    # served zero bytes (requests never reached uvicorn: no access
    # logs) across healthy containers, and Modal recycled one for
    # "initializing too long". Fresh APP_NAME also dodges the stale
    # routing of the thrice-deployed qwen38-ft name.
    from huggingface_hub import snapshot_download

    snap = snapshot_download(repo_id=BASE_MODEL, revision=BASE_REVISION)
    index = os.path.join(snap, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise RuntimeError(f"snapshot {snap} lacks the weight index")
    with open(index) as fh:
        weight_map = json.load(fh)["weight_map"]
    missing = sorted({f for f in weight_map.values()
                      if not os.path.isfile(os.path.join(snap, f))
                      or os.path.getsize(os.path.join(snap, f)) == 0})
    if missing:
        raise RuntimeError(f"snapshot {snap} incomplete: {missing[:5]}")
    print(f"serving base {BASE_MODEL}@{BASE_REVISION[:8]} "
          f"({len(set(weight_map.values()))} shards verified) + LoRA "
          f"{adapter} as {SERVED_NAME}", flush=True)
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
        # Same eval-critical flags as the working base serve: Qwen3.8
        # emits Qwen XML tool calls (hermes fails silently), and the
        # footage tasks send images.
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--limit-mm-per-prompt",
        '{"image": 8, "video": 2}',
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        # Mamba-block fix (2026-11-11): the hybrid tower needs one
        # Mamba cache block per decode seq; bf16 weights leave only
        # 280 blocks, so the 1024 default aborts cudagraph capture
        # ("lower max_num_seqs to at most 280"). 256 with margin.
        "--max-num-seqs",
        "256",
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
    # Popen-and-return, exactly like the working base serve: the
    # function returns while vLLM keeps serving the port.
    subprocess.Popen(cmd)
