"""Unit tests for hsea.models service-segment routing."""

from hsea.models import (
    BASETEN_BASE_URL,
    HF_BASE_URL,
    MODAL_CMDAPLUS_W4A4,
    baseten_model_id,
    hf_model_id,
    modal_model_id,
    required_env,
)


def test_baseten_model_id():
    assert baseten_model_id("meta-models/muse-spark-open-1026") == (
        "openai-api/baseten/meta-models/muse-spark-open-1026"
    )
    assert BASETEN_BASE_URL == "https://inference.baseten.co/v1"


def test_modal_model_id():
    assert modal_model_id("qwen") == "openai-api/modal/qwen"


def test_modal_cmdaplus_w4a4_routing():
    assert MODAL_CMDAPLUS_W4A4 == "CohereLabs/command-a-plus-05-2026-w4a4"
    assert modal_model_id(MODAL_CMDAPLUS_W4A4) == (
        "openai-api/modal/CohereLabs/command-a-plus-05-2026-w4a4"
    )


def test_hf_model_id():
    assert hf_model_id("deepseek-ai/DeepSeek-V4.1-Flash") == (
        "openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash"
    )
    assert HF_BASE_URL == "https://router.huggingface.co/v1"


def test_hf_model_id_provider_pin():
    assert hf_model_id("deepseek-ai/DeepSeek-V4.1-Flash", provider="deepinfra") == (
        "openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash:deepinfra"
    )


def test_required_env():
    assert required_env("baseten") == ("BASETEN_API_KEY", "BASETEN_BASE_URL")
    assert required_env("modal") == ("MODAL_API_KEY", "MODAL_BASE_URL")
    assert required_env("hf") == ("HF_API_KEY", "HF_BASE_URL")
