"""Tests for Monte Carlo null distribution and bootstrap CI."""

from dictcollision import bootstrap_ci, null_distribution


def test_null_distribution_shape():
    tokens = ["ab", "cd", "ef", "gh"] * 50
    dictionary = {"ab", "cd", "ef"}
    nd = null_distribution(tokens, dictionary, n=10)
    assert nd.n_samples == 10
    assert len(nd.net_signals) == 10
    pct = nd.observed_percentile()
    assert 0.0 <= pct <= 100.0


def test_bootstrap_ci_bracket_point():
    tokens = ["ab", "cd", "ef"] * 60
    dictionary = {"ab", "cd", "ef"}
    ci = bootstrap_ci(tokens, dictionary, n=30, confidence=0.90)
    assert ci.n_samples == 30
    # Sanity: point estimate should lie between lower and upper bound
    # (bootstrap percentile CI can be tight but typically encloses it).
    assert ci.lower <= ci.point_estimate + 1e-6
    assert ci.upper >= ci.point_estimate - 1e-6


def test_bootstrap_ci_empty_tokens():
    ci = bootstrap_ci([], {"a"}, n=5)
    assert ci.n_samples == 0
    assert ci.point_estimate == 0.0
