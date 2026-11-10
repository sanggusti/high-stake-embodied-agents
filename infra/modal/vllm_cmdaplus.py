"""Modal-hosted Command A+ W4A4 on 2xH100 behind an OpenAI-compatible endpoint (#5).

Same factory as vllm_qwen38.py: CUDA image + vLLM + shared HF/vLLM cache
volumes, served over HTTPS. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

Command A+ (``CohereLabs/command-a-plus-05-2026-w4a4``, 218B MoE / 25B
active, Apache-2.0, ungated, ~132 GB): the frontier reference pick
(docs/models.md shortlist pick #1). Serve recipe is the W4A4 model
card's vLLM section: vLLM >= 0.25.0, ``cohere_melody >= 0.9.0``,
``--tool-call-parser cohere_command4 --reasoning-parser cohere_command4
--enable-auto-tool-choice``; card's stated minimum footprint is
1xB200 or 2xH100 (``-tp 2`` here).

Only the MoE experts are NVFP4 (attention path stays full precision).
Weights download (~132 GB in 25 shards) dominates cold start; the
deploy uses the ``dmb-hf-token`` Modal secret (``HF_TOKEN``) so the
download is authenticated. Create it before deploying::

    modal secret create dmb-hf-token HF_TOKEN=<hf-read-token>
"""

import modal

MODEL_NAME = "CohereLabs/command-a-plus-05-2026-w4a4"
APP_NAME = "dmb-vllm-cmdaplus"
MODEL_REVISION: str | None = None  # pin a commit for reproducibility

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0", "cohere_melody>=0.9.0", "transformers")
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
    gpu="H100:2",
    secrets=[modal.Secret.from_name("dmb-hf-token")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/models": models_vol,
    },
    timeout=7200,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=3600)
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
        # Cohere-stated 2xH100 footprint for W4A4.
        "--tensor-parallel-size",
        "2",
        # Model-card recipe: cohere_command4 tool + reasoning parsers
        # (NOT cohere_command3 — that one is for command-a-reasoning).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "cohere_command4",
        "--reasoning-parser",
        "cohere_command4",
        # 132 GB weights on 2x80 GB leaves ~28 GB for KV/activations;
        # sliding-window attention (3:1) keeps KV small (~2.4 GB at 64K),
        # so 64K fits with headroom; native 128K stays for later.
        "--max-model-len",
        "65536",
        "--gpu-memory-utilization",
        "0.95",
        # forest_fire accumulates up to 8 snapshots in one prompt
        # (HSEA_EDGE_MAX_SNAPSHOTS_PER_EPISODE); match issue-1's budget.
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
