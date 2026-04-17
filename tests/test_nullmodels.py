"""Tests for unigram/bigram/trigram null models and classify integration."""

from dictcollision import classify, classify_by_length
from dictcollision._nullcorpus import generate_null_corpus


def test_unigram_preserves_lengths():
    tokens = ["abc", "defg", "hi", "jklmn"]
    null = generate_null_corpus(tokens, seed=0, null_model="unigram")
    assert [len(t) for t in null] == [len(t) for t in tokens]


def test_trigram_preserves_lengths():
    tokens = ["abcdef", "ghijkl", "mnopqr"] * 10
    null = generate_null_corpus(tokens, seed=0, null_model="trigram")
    assert [len(t) for t in null] == [len(t) for t in tokens]


def test_classify_accepts_null_model():
    tokens = ["hello", "world"] * 50
    dictionary = {"hello", "world"}
    for model in ("unigram", "bigram", "trigram"):
        r = classify(tokens, dictionary, n_nulls=3, null_model=model)
        assert r.apparent_hit_rate == 1.0


def test_classify_by_length_returns_buckets():
    tokens = ["ab", "cd", "abc", "def", "abcd"] * 40
    dictionary = {"ab", "cd", "abc", "def", "abcd"}
    buckets = classify_by_length(tokens, dictionary, n_nulls=3)
    lengths = {b.length for b in buckets}
    assert lengths == {2, 3, 4}
    for b in buckets:
        assert b.n_tokens > 0
        assert -1.0 <= b.net_signal <= 1.0
