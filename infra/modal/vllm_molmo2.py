"""Modal-hosted Molmo2-8B behind an OpenAI-compatible endpoint (issue #2).

Same factory shape as ``vllm_server.py`` but serving ``allenai/Molmo2-8B``
(Apache-2.0 weights, ungated; docs/models.md shortlist pick #3) — the
grounding VLM (pointing/grounding + multi-frame tracking) HSEA depends on
for "find the survivor in the drone feed" evals. inspect-ai reaches it as
``openai-api/modal/<model>`` with ``MODAL_BASE_URL=<serve_url>/v1``.

vLLM pin (nightly, per issue #2 contract + docs/models.md):
    vllm==0.34.0rc1.dev127+g8afb5e9ff
    wheel commit 8afb5e9ffeca2fd433f455c0e2b17814aea34c6d, built 2026-11-10
    (https://wheels.vllm.ai/nightly/ index of 2026-11-10T04:47 UTC).
Ai2 states Molmo2 needs vLLM >= 0.15.0 (github.com/allenai/molmo2), so
stable would likely serve — but the issue requires a nightly pin, and
nightly carries the latest Molmo2 multimodal fixes. Re-pin to a newer
nightly only after a verified deploy.

Tool calling: Molmo2's stock chat template has NO tool/system support
(roles must alternate user/assistant; a system message raises), while the
HSEA ReAct solver always prepends a system message and passes tools. So
serve() writes a Hermes-style template (CHAT_TEMPLATE) that keeps the
stock <|image|>/<|video|> prefix byte-identical and adds: system role
(backbone is Qwen3-family, im_start/im_end), a <tools> JSON block with a
Hermes preamble on the first system turn, assistant tool_calls rendered
as <tool_call>{"name","arguments"}</tool_call>, and tool results folded
into user turns as <tool_response> (Qwen3 convention). Parsed server-side
with --tool-call-parser hermes. Whether Molmo2 EMITS valid calls is the
issue #2 experiment — verify live, never assume.

Context: native 36864 (37K). An A10G (24 GB) OOMs at --max-model-len
16384 (EngineCore CUDA OOM, verified 2026-11-10 — same 8B-VLM wall as
Qwen3-VL-8B in docs/inference.md), so this serves on one A100-80GB with
--max-model-len 32768 (near-native; KV ~5 GB, weights ~16 GB BF16).
"""

import modal

MODEL_NAME = "allenai/Molmo2-8B"
APP_NAME = "dmb-vllm-molmo2"
MODEL_REVISION: str | None = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

# Pinned nightly wheel (see module docstring). Direct URL so the pin is
# exact — no floating --pre resolve at image-build time.
VLLM_NIGHTLY_URL = (
    "https://wheels.vllm.ai/"
    "8afb5e9ffeca2fd433f455c0e2b17814aea34c6d/vllm-0.34.0rc1.dev127"
    "%2Bg8afb5e9ff-cp38-abi3-manylinux_2_28_x86_64.whl"
)

# Hermes-style tool template on the stock Molmo2 skeleton. The mm-prefix
# block is copied verbatim from the stock chat_template.jinja
# (allenai/Molmo2-8B @ MODEL_REVISION; subtitle role dropped — the
# OpenAI API never emits it). No-tools user/assistant rendering is
# byte-identical to stock (verified locally with jinja2 before deploy).
# With 9 tool schemas in context the model lapses into narration unless
# the format is restated near generation time (curl probes 2026-11-10),
# so tool-enabled requests also get a MUST-start-with-<tool_call>
# instruction plus a one-shot arm->takeoff example in the system
# preamble, and a one-line reminder appended to every user turn.
CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if message['content'] is not string and message['content'] %}{% for content in message['content'] %}{% if content is mapping and (content.get('type') == 'image' or 'image' in content or 'image_url' in content) %}{% set image_count.value = image_count.value + 1 %}{% elif content is mapping and (content.get('type') == 'video' or 'video' in content or 'video_url' in content) %}{% set video_count.value = video_count.value + 1 %}{% endif %}{% endfor %}{% endif %}{% endfor %}{% if image_count.value == 1 %}{{ '<|image|>' }}{% elif image_count.value > 1 %}{% for i in range(image_count.value) %}{{ 'Image ' ~ (i + 1) ~ '<|image|>' }}{% endfor %}{% endif %}{% for _ in range(video_count.value) %}{{ '<|video|>' }}{% endfor %}{% set ns = namespace(closed=true, tools_rendered=false) %}{% for message in messages %}{% set role = message['role'] %}{% if role == 'system' %}{% if not ns.closed %}{{ '<|im_end|>\\n' }}{% endif %}{{ '<|im_start|>system\\n' }}{{ message['content'] }}{% if tools and not ns.tools_rendered %}

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{ tools | tojson(indent=2) }}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
IMPORTANT: to take an action, your response MUST start with a <tool_call> block. Plain-text narration changes nothing — never describe an action without its matching <tool_call> block in the same response.
Example — to start the drone, respond with ONLY:
<tool_call>
{"name": "arm", "arguments": {"drone_idx": 0}}
</tool_call>
After the tool result arrives, continue with the next call, e.g.:
<tool_call>
{"name": "takeoff", "arguments": {"drone_idx": 0, "altitude_m": 10}}
</tool_call>
When the mission is complete, submit using the submit function in the same <tool_call> format. Do not ask the user for confirmation; when you are ready, emit the <tool_call> block directly.{% set ns.tools_rendered = true %}{% endif %}{% set ns.closed = false %}{% elif role == 'tool' %}{% if message['content'] is string %}{% set tool_text = message['content'] %}{% elif message['content'] %}{% set t = namespace(text='') %}{% for content in message['content'] %}{% if content is mapping and content.get('type') == 'text' and content['text'] is string %}{% set t.text = t.text ~ ('' if not t.text else ' ') ~ content['text'] %}{% endif %}{% endfor %}{% set tool_text = t.text %}{% else %}{% set tool_text = '' %}{% endif %}{% if not ns.closed %}{{ '<|im_end|>\\n' }}{% endif %}{{ '<|im_start|>user\\n<tool_response>\\n' }}{{ tool_text }}{{ '\\n</tool_response>\\n<|im_end|>\\n' }}{% set ns.closed = true %}{% elif role == 'assistant' %}{% if not ns.closed %}{{ '<|im_end|>\\n' }}{% endif %}{{ ' <|im_start|>assistant\\n' }}{% if message['content'] is string %}{{ message['content'] }}{% elif message['content'] %}{% for content in message['content'] %}{% if content is mapping and content.get('type') == 'text' and content['text'] is string %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% if message.get('tool_calls') %}{% for tc in message['tool_calls'] %}{% set _args = tc['function']['arguments'] %}{{ '<tool_call>\\n{"name": "' ~ tc['function']['name'] ~ '", "arguments": ' ~ (_args if _args is string else _args | tojson) ~ '}\\n</tool_call>\\n' }}{% endfor %}{% endif %}{% set ns.closed = false %}{% else %}{% if message['content'] is string %}{% set text_content = message['content'] %}{% elif message['content'] %}{% set m = namespace(text='') %}{% for content in message['content'] %}{% if content is mapping and content.get('type') == 'text' and content['text'] is string %}{% set m.text = m.text ~ ('' if not m.text else ' ') ~ content['text'] %}{% endif %}{% endfor %}{% set text_content = m.text %}{% else %}{% set text_content = '' %}{% endif %}{% if not ns.closed %}{{ '<|im_end|>\\n' }}{% endif %}{{ '<|im_start|>user\\n' }}{{ text_content }}{% if tools %}{{ '\\n\\n(To act, start your response with a <tool_call> block using one of the <tools> above. Plain text changes nothing.)' }}{% endif %}{{ '<|im_end|>\\n' }}{% set ns.closed = true %}{% endif %}{% endfor %}{% if add_generation_prompt %}{% if not ns.closed %}{{ '<|im_end|>\\n' }}{% endif %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""  # noqa: E501

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

    # Custom tool template (see CHAT_TEMPLATE): vLLM takes a file path.
    with open("/tmp/molmo2-hermes.jinja", "w") as f:
        f.write(CHAT_TEMPLATE)

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        # Hermes tool parser + custom template (stock template cannot
        # render tools/system at all — see module docstring).
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--chat-template",
        "/tmp/molmo2-hermes.jinja",
        # Remote processor (processing_molmo2.py) on the official Ai2 repo.
        "--trust-remote-code",
        # Near-native 32768 on one A100-80GB (A10G OOMs at 16384, see
        # module docstring). Nightly vLLM forces --disable_chunked_mm_input
        # for Molmo2's bidirectional mm attention and reports
        # max_tokens_per_mm_item = max_model_len - 33, which exceeds the
        # default max_num_batched_tokens 2048 and aborts EngineCore — so
        # batch the full context (correct for single-user serving).
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        # Episodes accumulate up to snapshot-budget frames in one prompt.
        "--limit-mm-per-prompt",
        '{"image": 8}',
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
