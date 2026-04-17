"""Tests for the four-category classifier."""

from dictcollision import classify


def test_perfect_signal():
    # A long unique word dominates the corpus; bigram-generated nulls
    # almost never reproduce it exactly.
    tokens = ["antidisestablishmentarianism"] * 100
    dictionary = {"antidisestablishmentarianism"}
    result = classify(tokens, dictionary, n_nulls=5)
    assert result.signal > 0.5
    assert result.net_signal > 0.0


def test_empty_tokens():
    result = classify([], {"word"}, n_nulls=3)
    assert result.n_tokens == 0
    assert result.signal == 0.0


def test_empty_dictionary():
    tokens = ["abc", "def", "ghi"]
    result = classify(tokens, set(), n_nulls=3)
    assert result.signal == 0.0
    assert result.apparent_hit_rate == 0.0
    assert result.shared_miss > 0.0


def test_net_signal_is_signal_minus_anti():
    tokens = ["ab", "cd", "ef"] * 50
    dictionary = {"ab", "cd", "ef", "gh", "ij", "kl"}
    result = classify(tokens, dictionary, n_nulls=5)
    assert abs(result.net_signal - (result.signal - result.anti_signal)) < 1e-9


def test_apparent_is_signal_plus_shared_hit():
    tokens = ["the", "cat", "sat"] * 30
    dictionary = {"the", "cat", "sat", "mat"}
    result = classify(tokens, dictionary, n_nulls=5)
    assert abs(result.apparent_hit_rate - (result.signal + result.shared_hit)) < 1e-9


def test_pregenerated_nulls_are_used():
    tokens = ["hello", "world"] * 20
    dictionary = {"hello", "world"}
    # Empty nulls => no word shows up in nulls => no shared_hit, no anti_signal.
    nulls = [[], [], []]
    result = classify(tokens, dictionary, null_corpora=nulls)
    assert result.shared_hit == 0.0
    assert result.anti_signal == 0.0
    assert result.signal == 1.0


def test_correction_is_nonnegative():
    tokens = ["ab", "cd"] * 100
    dictionary = {"ab", "cd", "ef", "gh"}
    result = classify(tokens, dictionary, n_nulls=5)
    # apparent >= net when anti_signal + shared_hit contribute positively,
    # which they do whenever nulls produce any dict matches.
    assert result.correction == result.apparent_hit_rate - result.net_signal
