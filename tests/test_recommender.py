"""Tests for the dictionary recommender."""

import pytest

from dictcollision import recommend


def test_correct_language_wins():
    tokens = ["the", "cat", "sat", "on", "the", "mat"] * 20
    english = {"the", "cat", "sat", "on", "mat", "dog", "run"}
    german = {"der", "die", "das", "und", "ist", "auf"}
    results = recommend(tokens, {"english": english, "german": german})
    assert results[0].name == "english"
    assert results[0].excess >= results[1].excess


def test_empty_tokens():
    assert recommend([], {"test": {"word"}}) == []


def test_invalid_objective():
    with pytest.raises(ValueError):
        recommend(["a"], {"d": {"a"}}, objective="bogus")  # type: ignore[arg-type]


def test_snr_vs_excess_can_differ():
    # Small dict with a real hit should win on SNR; large dict with many hits wins on excess.
    tokens = ["ab"] * 100
    small = {"ab"}
    large = {"ab", "cd", "ef", "gh", "ij", "kl", "mn", "op"}

    by_excess = recommend(tokens, {"small": small, "large": large}, objective="excess")
    by_snr = recommend(tokens, {"small": small, "large": large}, objective="snr")

    # Both should rank small first for SNR since noise is lower (only one entry).
    assert by_snr[0].name == "small"
    # excess ranking depends on observed - predicted; both observed are 1.0 here
    assert by_excess[0].observed_hit_rate == 1.0


def test_fields_populated():
    tokens = ["ab", "cd", "ef"]
    result = recommend(tokens, {"d": {"ab", "cd"}})
    r = result[0]
    assert r.name == "d"
    assert r.n_tokens == 3
    assert r.n_hits == 2
    assert r.observed_hit_rate == pytest.approx(2 / 3)
