"""Unit tests for hsea.models service-segment routing."""

from hsea.models import (
    BASETEN_BASE_URL,
    HF_BASE_URL,
    MODAL_GEMMA4_12B,
    MODAL_QWEN3VL_8B,
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


def test_modal_gemma4_12b_routing():
    assert MODAL_GEMMA4_12B == "google/gemma-4-12B-it"
    assert modal_model_id(MODAL_GEMMA4_12B) == (
        "openai-api/modal/google/gemma-4-12B-it"
    )


def test_modal_qwen3vl_8b_routing():
    assert MODAL_QWEN3VL_8B == "Qwen/Qwen3-VL-8B-Instruct"
    assert modal_model_id(MODAL_QWEN3VL_8B) == (
        "openai-api/modal/Qwen/Qwen3-VL-8B-Instruct"
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
