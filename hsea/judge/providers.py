"""Judge providers: endpoints, rates, and cost math (issue-62 followup).

Pure config plus arithmetic (no API calls). The Meta Model API is
OpenAI-compatible at META_BASE_URL; the HF router pair is the pilot
reference. Rates are USD per 1M tokens, checked 2026-11-10.

Copy rules: no em-dashes in this module (commas, colons, hyphens only).
"""

from __future__ import annotations

META_BASE_URL = "https://api.meta.ai/v1"
HF_BASE_URL = "https://router.huggingface.co/v1"

# Per-1M-token rates: (input, output, cached_input). Reasoning tokens
# bill as output on the Meta API, which dominates judge cost at high
# effort. Sources: Meta Model API pricing via vendor docs mirrors,
# checked 2026-11-10 (links in docs/reports/issue-62-meta-judge.md).
META_RATES: dict[str, dict[str, float]] = {
    "muse-ember-1.0": {"input": 0.10, "output": 0.50,
                       "cached_input": 0.004},
    "muse-spark-1.1": {"input": 1.25, "output": 4.25,
                       "cached_input": 0.15},
    "muse-spark-1.2": {"input": 1.25, "output": 4.25,
                       "cached_input": 0.15},
    "muse-spark-1.3": {"input": 1.25, "output": 4.25,
                       "cached_input": 0.15},
    "muse-wonder-1.0": {"input": 4.00, "output": 20.00,
                        "cached_input": 0.20},
}

# Calibrated judge configs. max_tokens must cover reasoning PLUS the
# verdict text: wonder-low and ember-low emit 450-600 reasoning tokens
# on judge prompts, so a 600 cap truncates the verdict (probe evidence
# in the meta-judge report). minimal effort keeps ember under 200.
META_JUDGE_CONFIGS: dict[str, dict[str, object]] = {
    "wonder-low": {"model": "muse-wonder-1.0",
                   "reasoning_effort": "low", "max_tokens": 1200},
    "ember-low": {"model": "muse-ember-1.0",
                  "reasoning_effort": "low", "max_tokens": 1200},
    "ember-minimal": {"model": "muse-ember-1.0",
                      "reasoning_effort": "minimal", "max_tokens": 600},
    "spark-low": {"model": "muse-spark-1.3",
                  "reasoning_effort": "low", "max_tokens": 1200},
}

# Pilot reference judges (HF router). v41-flash rates observed
# 2026-11-09 per docs/inference.md; gemma had no listed pair, so the
# pilot actuals ($/judgment) are the reference, not a rate pair.
HF_JUDGE_A = "deepseek-ai/DeepSeek-V4.1-Flash"
HF_JUDGE_B = "google/gemma-3-27b-it"
HF_RATE_IN = 0.30
HF_RATE_OUT = 1.20


def judgment_cost(prompt_tokens: int, completion_tokens: int,
                  model: str) -> float:
    """Cost of one call in USD (reasoning bills as output)."""
    if model not in META_RATES:
        raise ValueError(f"unknown model {model!r}")
    rates = META_RATES[model]
    return (prompt_tokens * rates["input"]
            + completion_tokens * rates["output"]) / 1_000_000


def per_1k_cost(mean_in: float, mean_out: float, model: str) -> float:
    """Projected USD per 1,000 judgments at observed mean tokens."""
    return judgment_cost(mean_in, mean_out, model) * 1000


def config_max_tokens(config: str) -> int:
    """Completion budget for a calibrated judge config."""
    if config not in META_JUDGE_CONFIGS:
        raise ValueError(f"unknown judge config {config!r}")
    return int(META_JUDGE_CONFIGS[config]["max_tokens"])


def cheapest_first(configs: list[str]) -> list[str]:
    """Sort configs by output rate (input rates rank the same way)."""
    return sorted(configs,
                  key=lambda c: META_RATES[
                      str(META_JUDGE_CONFIGS[c]["model"])]["output"])
