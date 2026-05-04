"""Tests for summary methods on result types."""

from dictcollision import classify, recommend
from dictcollision._types import BootstrapCI, NullDistribution


def test_classify_result_summary_contains_key_fields():
    tokens = ["the", "cat"] * 20
    dictionary = {"the", "cat"}
    r = classify(tokens, dictionary, n_nulls=3)
    s = r.summary()
    assert "net signal" in s
    assert "apparent hit rate" in s
    assert "Interpretation" in s


def test_short_text_summary_includes_search_calibration_hint():
    # n=40 < 200 -> warning fires.
    tokens = ["abcd", "efgh"] * 20
    dictionary = {"abcd"}
    r = classify(tokens, dictionary, n_nulls=3)
    s = r.summary()
    assert "search_calibrated_signal" in s
    assert "n=40" in s


def test_long_text_summary_omits_search_calibration_hint():
    # n=300 > 200 -> warning suppressed.
    tokens = ["xy", "zw"] * 150
    dictionary = {"xy"}
    r = classify(tokens, dictionary, n_nulls=3)
    s = r.summary()
    assert "search_calibrated_signal" not in s


def test_classify_result_to_dict_serializable():
    import json

    r = classify(["a", "b"], {"a"}, n_nulls=3)
    d = r.to_dict()
    json.dumps(d)  # must be JSON-serializable


def test_recommendation_summary_line():
    tokens = ["ab", "cd"] * 20
    ranked = recommend(tokens, {"d": {"ab", "cd"}})
    s = ranked[0].summary()
    assert "observed" in s
    assert "excess" in s


def test_null_distribution_percentile():
    nd = NullDistribution(
        net_signals=[-0.1, -0.05, 0.0, 0.05, 0.1],
        observed_net_signal=0.2,
        n_samples=5,
    )
    assert nd.observed_percentile() == 100.0
    assert nd.percentile_of(-0.2) == 0.0
    assert 30.0 < nd.percentile_of(0.0) < 70.0


def test_bootstrap_ci_summary():
    ci = BootstrapCI(
        point_estimate=0.3,
        lower=0.25,
        upper=0.35,
        confidence=0.95,
        n_samples=100,
    )
    s = ci.summary()
    assert "30.0%" in s
    assert "95%" in s
