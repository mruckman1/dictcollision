"""Tests for the core collision equation."""

from dictcollision import character_frequencies, noise_floor, token_length_distribution


def test_empty_tokens():
    assert noise_floor([], ["word"]) == 0.0


def test_empty_dictionary():
    assert noise_floor(["abc", "def"], []) == 0.0


def test_known_collision():
    # Tokens "aa", "bb": char freq of 'a' is 0.5, 'b' is 0.5.
    # Dictionary ["aa"]: P(match "aa") = 0.5 * 0.5 = 0.25.
    # With 2 tokens both of length 2, noise floor = 0.25.
    tokens = ["aa", "bb"]
    dictionary = ["aa"]
    result = noise_floor(tokens, dictionary)
    assert abs(result - 0.25) < 0.01


def test_impossible_characters():
    # Dictionary has characters not in decoded tokens -> 0 contribution.
    tokens = ["abc", "def"]
    dictionary = ["xyz"]
    result = noise_floor(tokens, dictionary)
    assert result == 0.0


def test_character_frequencies_basic():
    freqs = character_frequencies(["ab", "ab"])
    assert abs(freqs["a"] - 0.5) < 0.01
    assert abs(freqs["b"] - 0.5) < 0.01


def test_character_frequencies_empty():
    assert character_frequencies([]) == {}
    assert character_frequencies([""]) == {}


def test_token_length_distribution():
    dist = token_length_distribution(["a", "bb", "ccc", "dd"])
    assert dist == {1: 1, 2: 2, 3: 1}


def test_noise_floor_scales_with_dict_size():
    # Larger dictionary with random entries -> higher collision rate.
    import random

    rng = random.Random(0)
    chars = "abcdefghij"
    tokens = ["".join(rng.choice(chars) for _ in range(3)) for _ in range(200)]

    small = [rng.choice(chars) + rng.choice(chars) + rng.choice(chars) for _ in range(5)]
    large = [rng.choice(chars) + rng.choice(chars) + rng.choice(chars) for _ in range(200)]

    small_noise = noise_floor(tokens, small)
    large_noise = noise_floor(tokens, large)
    assert large_noise >= small_noise


def test_noise_floor_in_unit_interval():
    tokens = ["ab"] * 10
    dictionary = ["ab", "ab", "ab"]  # duplicates allowed in iterable
    result = noise_floor(tokens, dictionary)
    assert 0.0 <= result <= 1.0


def test_char_freqs_override():
    # Override character frequencies
    tokens = ["aa"]
    dictionary = ["aa"]
    # With forced uniform 0.1 freqs, P(aa) = 0.01
    freqs = {"a": 0.1}
    result = noise_floor(tokens, dictionary, char_freqs=freqs)
    assert abs(result - 0.01) < 1e-9
