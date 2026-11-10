"""Model routing for inspect-ai across inference providers.

All providers expose OpenAI-compatible endpoints, so all are addressed
through inspect-ai's ``openai-api/<service>/<model>`` interface. Verified
against the installed inspect-ai provider implementation: the service segment
selects ``<SERVICE>_API_KEY`` and ``<SERVICE>_BASE_URL`` (upper-cased,
dashes become underscores).
"""

from __future__ import annotations

BASETEN_BASE_URL = "https://inference.baseten.co/v1"
HF_BASE_URL = "https://router.huggingface.co/v1"

#: Modal-hosted SmolLM3-3B (issue #4; see
#: ``infra/modal/vllm_smollm3.py``). Full inspect-ai id is
#: ``modal_model_id(MODAL_SMOLLM3_3B)``.
MODAL_SMOLLM3_3B = "HuggingFaceTB/SmolLM3-3B"

#: Modal-hosted SmolVLM-256M-Instruct (issue #4; see
#: ``infra/modal/vllm_smolvlm.py``). Full inspect-ai id is
#: ``modal_model_id(MODAL_SMOLVLM_256M)``.
MODAL_SMOLVLM_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"

#: Modal-hosted Tiny Aya L2 Thinker (issue #4; see
#: ``infra/modal/vllm_tinyaya.py``). CC-BY-NC — research-only. Full
#: inspect-ai id is ``modal_model_id(MODAL_TINYAYA_L2)``.
MODAL_TINYAYA_L2 = "CohereLabs/tiny-aya-l2-thinker"


def baseten_model_id(model_slug: str) -> str:
    """inspect-ai model id for a Baseten Model APIs slug.

    Requires ``BASETEN_API_KEY`` and ``BASETEN_BASE_URL`` in the environment.
    """
    return f"openai-api/baseten/{model_slug}"


def modal_model_id(model_name: str) -> str:
    """inspect-ai model id for a Modal-hosted vLLM deployment.

    Requires ``MODAL_API_KEY`` and ``MODAL_BASE_URL`` (the deployment's
    ``.../v1`` URL) in the environment.
    """
    return f"openai-api/modal/{model_name}"


def hf_model_id(model_slug: str, provider: str | None = None) -> str:
    """inspect-ai model id for a Hugging Face Inference Providers model.

    Requires ``HF_API_KEY`` (a user access token with Inference Providers
    permission) and ``HF_BASE_URL`` (``HF_BASE_URL`` constant) in the
    environment. ``model_slug`` is the Hub repo id (e.g.
    ``deepseek-ai/DeepSeek-V4.1-Flash``); pass ``provider`` to pin a
    serving backend (``model:provider`` router syntax, e.g. ``deepinfra``).
    """
    if provider:
        model_slug = f"{model_slug}:{provider}"
    return f"openai-api/hf/{model_slug}"


def required_env(service: str) -> tuple[str, ...]:
    """Environment variables inspect-ai reads for a service segment."""
    prefix = service.upper().replace("-", "_")
    return (f"{prefix}_API_KEY", f"{prefix}_BASE_URL")
