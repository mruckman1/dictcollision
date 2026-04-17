"""Tests for the paper's baseline correction methods."""

from dictcollision.baselines import (
    all_methods,
    apparent_hit_rate,
    bh_fdr,
    blast_evalue,
    permutation_test,
    subtract_null,
)


def test_apparent_hit_rate_matches_simple_count():
    tokens = ["a", "b", "a", "c"]
    assert apparent_hit_rate(tokens, {"a"}) == 0.5


def test_subtract_null_nonzero_on_signal():
    tokens = ["hello", "world"] * 50
    # Real hit rate is 1.0; nulls rarely produce these 5-char words.
    val = subtract_null(tokens, {"hello", "world"}, n_nulls=3)
    assert val > 0.5


def test_all_methods_returns_all_keys():
    tokens = ["ab", "cd"] * 50
    keys = all_methods(tokens, {"ab", "cd"}, n_nulls=3)
    assert set(keys) == {
        "apparent_hit_rate",
        "subtract_null",
        "permutation_test",
        "bh_fdr",
        "blast_evalue",
        "four_category_net",
    }


def test_empty_tokens():
    assert apparent_hit_rate([], {"a"}) == 0.0
    assert subtract_null([], {"a"}) == 0.0
    assert permutation_test([], {"a"}) == 0.0
    assert bh_fdr([], {"a"}) == 0.0
    assert blast_evalue([], {"a"}) == 0.0


def test_methods_in_unit_interval_or_close():
    # All methods should produce reasonable values; single-word tests can
    # exceed apparent because they accept word TYPES whose weighted count
    # summed equals apparent hit rate at most.
    tokens = ["ab", "cd", "ef", "gh", "ij"] * 20
    dictionary = {"ab", "cd", "ef"}
    methods = all_methods(tokens, dictionary, n_nulls=3)
    for name, val in methods.items():
        # Reasonable range: [-1, 1]
        assert -1.0 <= val <= 1.0, f"{name} out of range: {val}"
