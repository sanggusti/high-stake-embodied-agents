"""Meta judge provider config and cost math (offline, no API)."""

import pytest

from hsea.judge import providers as P


def test_rates_cover_catalog():
    for model in ("muse-ember-1.0", "muse-spark-1.1", "muse-spark-1.2",
                  "muse-spark-1.3", "muse-wonder-1.0"):
        rates = P.META_RATES[model]
        assert rates["input"] > 0 and rates["output"] > 0
        assert rates["cached_input"] < rates["input"]


def test_judgment_cost_bills_reasoning_as_output():
    # 1229 in / 623 out (472 reasoning) on wonder: probe actual.
    cost = P.judgment_cost(1229, 623, "muse-wonder-1.0")
    assert cost == pytest.approx((1229 * 4.00 + 623 * 20.00) / 1e6)
    assert cost == pytest.approx(0.017376, abs=1e-6)


def test_per_1k_ordering_ember_spark_wonder():
    mean_in, mean_out = 1415.0, 469.0
    ember = P.per_1k_cost(mean_in, mean_out, "muse-ember-1.0")
    spark = P.per_1k_cost(mean_in, mean_out, "muse-spark-1.3")
    wonder = P.per_1k_cost(mean_in, mean_out, "muse-wonder-1.0")
    assert ember < spark < wonder
    assert ember == pytest.approx(0.376, abs=0.01)
    assert wonder == pytest.approx(15.04, abs=0.05)


def test_low_effort_budgets_cover_reasoning_plus_verdict():
    # Low effort emits 450-600 reasoning tokens on judge prompts, so
    # the cap must clear 600 plus ~200 tokens of verdict text.
    assert P.config_max_tokens("wonder-low") >= 800
    assert P.config_max_tokens("ember-low") >= 800
    assert P.config_max_tokens("ember-minimal") == 600


def test_unknown_model_and_config_raise():
    with pytest.raises(ValueError, match="unknown model"):
        P.judgment_cost(1, 1, "muse-nope-9.9")
    with pytest.raises(ValueError, match="unknown judge config"):
        P.config_max_tokens("wonder-ultra")


def test_cheapest_first_ranks_ember_lowest():
    ranked = P.cheapest_first(["wonder-low", "spark-low",
                               "ember-low", "ember-minimal"])
    assert ranked[0] in ("ember-low", "ember-minimal")
    assert ranked[-1] == "wonder-low"
