"""Tests for null corpus generation."""

from dictcollision._nullcorpus import generate_null_corpora, generate_null_corpus


def test_empty_tokens():
    assert generate_null_corpus([]) == []


def test_length_distribution_preserved():
    tokens = ["ab", "cde", "fghi", "jk", "lm"]
    null = generate_null_corpus(tokens, seed=0)
    assert [len(t) for t in null] == [len(t) for t in tokens]


def test_reproducibility():
    tokens = ["abc"] * 50
    a = generate_null_corpus(tokens, seed=7)
    b = generate_null_corpus(tokens, seed=7)
    assert a == b


def test_different_seeds_differ():
    tokens = ["abc", "def", "ghi", "jkl"] * 20
    a = generate_null_corpus(tokens, seed=1)
    b = generate_null_corpus(tokens, seed=2)
    assert a != b


def test_generate_multiple():
    tokens = ["abc", "def", "ghi"] * 10
    corpora = generate_null_corpora(tokens, n=5, base_seed=0)
    assert len(corpora) == 5
    for c in corpora:
        assert len(c) == len(tokens)


def test_uses_only_observed_characters():
    tokens = ["ab", "ba"]
    null = generate_null_corpus(tokens, seed=0)
    observed_chars = set("ab")
    for tok in null:
        assert set(tok) <= observed_chars
