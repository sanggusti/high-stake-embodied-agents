"""Unit tests for hsea.models service-segment routing."""

from hsea.models import (
    BASETEN_BASE_URL,
    HF_BASE_URL,
    MODAL_SMOLLM3_3B,
    MODAL_SMOLVLM_256M,
    MODAL_TINYAYA_L2,
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


def test_modal_edge_lane_routing():
    assert MODAL_SMOLLM3_3B == "HuggingFaceTB/SmolLM3-3B"
    assert modal_model_id(MODAL_SMOLLM3_3B) == (
        "openai-api/modal/HuggingFaceTB/SmolLM3-3B"
    )
    assert MODAL_SMOLVLM_256M == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert modal_model_id(MODAL_SMOLVLM_256M) == (
        "openai-api/modal/HuggingFaceTB/SmolVLM-256M-Instruct"
    )
    assert MODAL_TINYAYA_L2 == "CohereLabs/tiny-aya-l2-thinker"
    assert modal_model_id(MODAL_TINYAYA_L2) == (
        "openai-api/modal/CohereLabs/tiny-aya-l2-thinker"
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
