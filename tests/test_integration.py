"""End-to-end integration test replicating the paper's core finding."""

import random

from dictcollision import classify, noise_floor


def test_paper_core_finding():
    """Short tokens + large dictionary = high noise floor."""
    rng = random.Random(42)

    # Small alphabet and enough tokens so each type is repeated across
    # real and null corpora — this is the regime where the classifier
    # can separate signal from shared_hit.
    chars = "abcdefgh"
    tokens = [rng.choice(chars) + rng.choice(chars) for _ in range(5000)]

    # Small dictionary: low noise.
    small_dict = [c1 + c2 for c1 in "ab" for c2 in "ab"]  # 4 entries
    small_noise = noise_floor(tokens, small_dict)

    # Large dictionary: covers all 2-char combos -> ~100% noise.
    large_dict = [c1 + c2 for c1 in chars for c2 in chars]  # 64 entries
    large_noise = noise_floor(tokens, large_dict)

    # Collision effect: large dict produces much more noise.
    assert large_noise > small_noise * 10

    # With a complete 2-char dictionary, noise should be ~100%.
    assert large_noise > 0.95

    # Four-category should catch this: random tokens against full dict
    # produce no real signal — every type also shows up in nulls.
    result = classify(tokens, set(large_dict), n_nulls=5)
    assert result.net_signal < 0.05
    assert result.shared_hit > 0.9  # most tokens are shared_hit (chance)


def test_long_tokens_have_little_noise():
    """Long unique tokens rarely collide with dictionary entries."""
    rng = random.Random(1)
    chars = "abcdefghijklmnopqrstuvwxyz"
    tokens = ["".join(rng.choice(chars) for _ in range(12)) for _ in range(200)]

    # 1000-entry dictionary of length-12 strings (unrelated to tokens)
    dictionary = [
        "".join(rng.choice(chars) for _ in range(12)) for _ in range(1000)
    ]
    assert noise_floor(tokens, dictionary) < 0.01
