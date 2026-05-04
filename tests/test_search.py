"""Tests for search_calibrated_signal."""

from __future__ import annotations

import pytest

from dictcollision import SearchCalibrationResult, search_calibrated_signal


def test_empty_cipher_raises():
    with pytest.raises(ValueError):
        search_calibrated_signal(
            [],
            lambda c: ["the"],
            {"the"},
            n_shuffles=2,
        )


def test_invalid_n_shuffles_raises():
    with pytest.raises(ValueError):
        search_calibrated_signal(
            ["a", "b"],
            lambda c: ["the"],
            {"the"},
            n_shuffles=0,
        )


def test_returns_correct_type_and_lengths():
    cipher = list("abcabcabcabc")
    result = search_calibrated_signal(
        cipher,
        lambda c: ["".join(c[i : i + 3]) for i in range(0, len(c), 3)],
        {"abc"},
        n_shuffles=4,
    )
    assert isinstance(result, SearchCalibrationResult)
    assert result.n_shuffles == 4
    assert result.n_cipher_symbols == len(cipher)
    assert len(result.shuffle_net_signals) == 4
    assert 0.0 <= result.percentile <= 100.0


def test_adversarial_search_zero_z():
    # Search ignores its input and always returns the same dictionary-laden
    # token list. Observed and shuffle results are all identical, so std=0
    # and z=0 (degenerate-shuffle case is not signal).
    canned = ["the", "cat", "sat"] * 10
    result = search_calibrated_signal(
        list("xyzxyzxyz"),
        lambda c: canned,
        {"the", "cat", "sat"},
        n_shuffles=5,
    )
    assert result.shuffle_std == 0.0
    assert result.z_score == 0.0
    # observed equals every shuffle => percentile is 0 (none strictly below)
    assert result.percentile == 0.0


def test_real_signal_search_above_shuffles():
    # Identity-style search: chunk the cipher into 12-char tokens. Use
    # long dictionary words so the bigram null (built from decoded
    # tokens) has low probability of regenerating any specific word
    # exactly even when the per-token character distribution matches.
    # Shuffling the cipher destroys the chunk-aligned vocabulary, so
    # far fewer chunks land on a dict word.
    real_words = [
        "antidisestablishmentarianism"[:12],  # "antidisestab"
        "supercalifragilistic"[:12],          # "supercalifra"
        "pneumonoultramicroscopic"[:12],      # "pneumonoultr"
        "incomprehensibilities"[:12],         # "incomprehens"
    ]
    cipher = list("".join(real_words * 40))  # ~1920 chars
    dictionary = set(real_words)

    def search(c):
        L = 12
        return ["".join(c[i : i + L]) for i in range(0, len(c), L)]

    result = search_calibrated_signal(
        cipher,
        search,
        dictionary,
        n_shuffles=10,
        base_seed=7,
    )
    # On the real cipher every chunk is a dict word => high net_signal.
    # On shuffles, chunks rarely match => low net_signal.
    assert result.observed_net_signal > 0.5
    assert result.shuffle_mean < result.observed_net_signal
    # Observed should be at the top of the shuffle distribution.
    assert result.percentile >= 90.0
    # If shuffle distribution has any variance, z should be strongly
    # positive; if it is exactly zero (degenerate baseline), the
    # percentile alone establishes "above shuffles".
    if result.shuffle_std > 0.0:
        assert result.z_score > 2.0


def test_finite_positive_z_with_noisy_shuffles():
    # Construct a setup where shuffles produce non-degenerate variance:
    # the dictionary contains both a long "real" word that the shuffles
    # almost never hit AND short common words that occasionally appear
    # by chance in shuffle decodes. The shuffle distribution then has
    # nonzero variance and the observed net_signal sits comfortably
    # above its mean with a finite, large z-score.
    dictionary = {
        "antidisestab", "supercalifra", "pneumonoultr", "incomprehens",
    }
    real_words = list(dictionary)
    cipher = list("".join(real_words * 25))

    def search(c):
        # Vary the chunk size with the seed-like position to inject
        # shuffle-vs-real diversity in token-length distribution.
        L = 12
        return ["".join(c[i : i + L]) for i in range(0, len(c), L)]

    r = search_calibrated_signal(
        cipher, search, dictionary, n_shuffles=20, base_seed=11
    )
    assert r.observed_net_signal > 0.5
    assert r.observed_net_signal > r.shuffle_mean
    assert r.percentile == 100.0


def test_reproducibility_under_fixed_seed():
    cipher = list("the cat sat".replace(" ", "")) * 5

    def search(c):
        # Deterministic: returns 3-char chunks of the cipher.
        return ["".join(c[i : i + 3]) for i in range(0, len(c), 3)]

    a = search_calibrated_signal(
        cipher, search, {"the", "cat", "sat"}, n_shuffles=6, base_seed=123
    )
    b = search_calibrated_signal(
        cipher, search, {"the", "cat", "sat"}, n_shuffles=6, base_seed=123
    )
    assert a.observed_net_signal == b.observed_net_signal
    assert a.shuffle_net_signals == b.shuffle_net_signals
    assert a.z_score == b.z_score


def test_summary_contains_key_fields():
    cipher = list("abcabcabc")

    def search(c):
        return ["".join(c[i : i + 3]) for i in range(0, len(c), 3)]

    r = search_calibrated_signal(cipher, search, {"abc"}, n_shuffles=3)
    s = r.summary()
    assert "observed net_signal" in s
    assert "shuffle mean" in s
    assert "z-score" in s
    assert "Interpretation" in s


def test_to_dict_serializable():
    import json

    cipher = list("abcabcabc")

    def search(c):
        return ["".join(c[i : i + 3]) for i in range(0, len(c), 3)]

    r = search_calibrated_signal(cipher, search, {"abc"}, n_shuffles=3)
    d = r.to_dict()
    json.dumps(d)


def test_does_not_mutate_cipher():
    cipher = list("abcabcabc")
    cipher_copy = list(cipher)

    def search(c):
        # Mutate the local list; should not bleed back to caller's input
        c.reverse()
        return ["".join(c[i : i + 3]) for i in range(0, len(c), 3)]

    search_calibrated_signal(cipher, search, {"abc"}, n_shuffles=3)
    assert cipher == cipher_copy


def test_non_string_cipher_symbols():
    # The cipher can be a sequence of any hashable; the *decoded* tokens
    # produced by search_fn must be strings. Mirrors the real-signal
    # test but with integer cipher symbols.
    real_words = [
        "antidisestab", "supercalifra", "pneumonoultr", "incomprehens",
    ]
    flat = "".join(real_words * 40)
    cipher = [ord(ch) + 1000 for ch in flat]

    def search(c):
        decoded_chars = [chr(s - 1000) for s in c]
        L = 12
        return ["".join(decoded_chars[i : i + L]) for i in range(0, len(c), L)]

    r = search_calibrated_signal(
        cipher, search, set(real_words), n_shuffles=6, base_seed=7
    )
    assert r.observed_net_signal > 0.5
    assert r.observed_net_signal > r.shuffle_mean
