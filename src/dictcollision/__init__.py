"""
dictcollision -- Calibrate dictionary hit rates in computational decipherment.

Your decipherment reports a 43% dictionary hit rate. Is that real?

    >>> from dictcollision import noise_floor, classify, recommend

Quick check (one line, no null corpora needed):

    >>> predicted = noise_floor(decoded_tokens, dictionary)
    >>> print(f"Chance alone predicts {predicted:.1%}")

Full four-category analysis:

    >>> result = classify(decoded_tokens, dictionary)
    >>> print(f"Net signal: {result.net_signal:.1%}")

Rank multiple dictionaries:

    >>> ranked = recommend(tokens, {"latin_10k": lat, "german_50k": de})
    >>> print(ranked[0].name, ranked[0].excess)

Based on: Ruckman (2026), "The Dictionary Collision Effect
in Computational Decipherment."
"""

__version__ = "0.1.1"

from dictcollision._collision import (
    character_frequencies,
    noise_floor,
    token_length_distribution,
)
from dictcollision._framework import classify
from dictcollision._recommender import recommend
from dictcollision._types import ClassifyResult, Recommendation

__all__ = [
    "noise_floor",
    "classify",
    "recommend",
    "character_frequencies",
    "token_length_distribution",
    "ClassifyResult",
    "Recommendation",
    "__version__",
]
