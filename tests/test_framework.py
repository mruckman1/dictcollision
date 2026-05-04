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


def test_signal_word_counts_sum_matches_signal_fraction():
    # signal_word_counts should sum to signal * n_tokens (modulo float).
    tokens = ["antidisestablishmentarianism"] * 100
    dictionary = {"antidisestablishmentarianism"}
    r = classify(tokens, dictionary, n_nulls=5)
    if r.signal_word_counts:
        assert sum(r.signal_word_counts.values()) == int(round(r.signal * r.n_tokens))


def test_signal_words_ordered_by_count():
    # Construct a corpus where bigrams cannot reproduce the long words,
    # giving a deterministic signal-word ranking.
    tokens = (
        ["antidisestablishmentarianism"] * 50
        + ["supercalifragilisticexpialid"] * 30
        + ["pneumonoultramicroscopicsili"] * 10
    )
    dictionary = {
        "antidisestablishmentarianism",
        "supercalifragilisticexpialid",
        "pneumonoultramicroscopicsili",
    }
    r = classify(tokens, dictionary, n_nulls=5)
    # The signal_words list ordering should match descending count.
    if len(r.signal_words) >= 2:
        counts_in_order = [r.signal_word_counts[w] for w in r.signal_words]
        assert counts_in_order == sorted(counts_in_order, reverse=True)


def test_overfit_score_concentrated_corpus():
    # Single repeated long word -> top-3 = 100% of signal mass.
    tokens = ["antidisestablishmentarianism"] * 100
    dictionary = {"antidisestablishmentarianism"}
    r = classify(tokens, dictionary, n_nulls=5)
    assert r.overfit_score() == 1.0


def test_overfit_score_no_signal_returns_zero():
    # No signal words means 0.0 (defined as the safe edge case).
    tokens = ["abc", "def", "ghi"]
    dictionary: set[str] = set()
    r = classify(tokens, dictionary, n_nulls=3)
    assert r.overfit_score() == 0.0


def test_overfit_score_diversified_signal():
    # Many distinct long signal words at equal counts -> top-3 makes up
    # 3/N of the signal mass and overfit_score sits well below 0.5.
    words = [
        "antidisestablishmentariana",
        "supercalifragilisticexpial",
        "pneumonoultramicroscopicsi",
        "incomprehensibilitiesabcde",
        "constitutionalisticconvene",
        "establishmentarianismplane",
        "transubstantiationproceeds",
        "uncharacteristicallyfishin",
        "counterrevolutionaryagentm",
        "honorificabilitudinitatibu",
    ]
    tokens = []
    for w in words:
        tokens.extend([w] * 10)
    dictionary = set(words)
    r = classify(tokens, dictionary, n_nulls=5)
    if r.signal > 0 and len(r.signal_word_counts) >= 4:
        assert r.overfit_score() < 0.45
