"""
dictcollision -- Calibrate dictionary hit rates: separate real matches
from chance collisions.

The package answers one question: you have a list of short strings and
a big reference dictionary; some fraction match. How many are real
matches vs. the dictionary being big enough that anything would match?

Top-level API
-------------

Quick check:
    >>> from dictcollision import noise_floor
    >>> noise_floor(tokens, dictionary)            # one-line prediction

Four-category classification:
    >>> from dictcollision import classify
    >>> r = classify(tokens, dictionary)
    >>> print(r.summary())

Length-stratified breakdown:
    >>> from dictcollision import classify_by_length
    >>> buckets = classify_by_length(tokens, dictionary)

Monte Carlo null distribution and bootstrap CI:
    >>> from dictcollision import null_distribution, bootstrap_ci
    >>> null_distribution(tokens, dictionary, n=40)
    >>> bootstrap_ci(tokens, dictionary, n=200)

Dictionary recommender:
    >>> from dictcollision import recommend
    >>> recommend(tokens, {"latin_10k": lat, "german_50k": de})

Paper baselines (Table 2 / Figure 5 reproduction):
    >>> from dictcollision.baselines import all_methods
    >>> all_methods(tokens, dictionary)

File loaders:
    >>> from dictcollision import load_dictionary, load_tokens

Based on: Ruckman (2026), "The Dictionary Collision Effect
in Computational Decipherment." See github.com/mruckman1/signal-isolation-paper.
"""

__version__ = "0.2.0"

from dictcollision._collision import (
    character_frequencies,
    noise_floor,
    token_length_distribution,
)
from dictcollision._framework import classify, classify_by_length
from dictcollision._io import load_dictionary, load_tokens
from dictcollision._recommender import recommend
from dictcollision._stats import bootstrap_ci, null_distribution
from dictcollision._types import (
    BootstrapCI,
    ClassifyResult,
    LengthBucket,
    NullDistribution,
    Recommendation,
)

__all__ = [
    # Core API
    "noise_floor",
    "classify",
    "classify_by_length",
    "recommend",
    "null_distribution",
    "bootstrap_ci",
    # I/O
    "load_dictionary",
    "load_tokens",
    # Utility
    "character_frequencies",
    "token_length_distribution",
    # Result types
    "ClassifyResult",
    "Recommendation",
    "LengthBucket",
    "NullDistribution",
    "BootstrapCI",
    # Version
    "__version__",
]
