"""Model routing for inspect-ai across inference providers.

Both providers expose OpenAI-compatible endpoints, so both are addressed
through inspect-ai's ``openai-api/<service>/<model>`` interface. Verified
against the installed inspect-ai provider implementation: the service segment
selects ``<SERVICE>_API_KEY`` and ``<SERVICE>_BASE_URL`` (upper-cased,
dashes become underscores).
"""

from __future__ import annotations

BASETEN_BASE_URL = "https://inference.baseten.co/v1"


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


def required_env(service: str) -> tuple[str, ...]:
    """Environment variables inspect-ai reads for a service segment."""
    prefix = service.upper().replace("-", "_")
    return (f"{prefix}_API_KEY", f"{prefix}_BASE_URL")
