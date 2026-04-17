"""Generate null corpora from character bigram statistics.

A null corpus preserves the character-pair frequencies and token-length
distribution of the real tokens while destroying word identity. It is
the reference distribution against which the four-category classifier
measures chance collisions.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence


def _build_bigram_model(
    tokens: Sequence[str],
) -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
    """Build a first-order Markov model from character bigrams.

    Returns the transition table (cumulative probabilities for sampling)
    and a list of start characters weighted by frequency.
    """
    bigram_counts: dict[str, Counter[str]] = {}
    start_chars: list[str] = []

    for tok in tokens:
        if not tok:
            continue
        start_chars.append(tok[0])
        for i in range(len(tok) - 1):
            c1, c2 = tok[i], tok[i + 1]
            if c1 not in bigram_counts:
                bigram_counts[c1] = Counter()
            bigram_counts[c1][c2] += 1

    transitions: dict[str, list[tuple[str, float]]] = {}
    for c1, counts in bigram_counts.items():
        total = sum(counts.values())
        cumulative: list[tuple[str, float]] = []
        running = 0.0
        for c2, n in counts.items():
            running += n / total
            cumulative.append((c2, running))
        transitions[c1] = cumulative

    return transitions, start_chars


def _sample_char(table: list[tuple[str, float]], rng: random.Random) -> str:
    """Sample from a cumulative probability table."""
    r = rng.random()
    for char, cum_prob in table:
        if r <= cum_prob:
            return char
    return table[-1][0]


def generate_null_corpus(
    tokens: list[str],
    seed: int = 42,
) -> list[str]:
    """Generate one null corpus matching the character bigram statistics
    and token-length distribution of the input.

    Parameters
    ----------
    tokens : list of str
        Real decoded tokens to mimic.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Null tokens with same length distribution, character bigram
        frequencies preserved, word identity destroyed.
    """
    if not tokens:
        return []

    rng = random.Random(seed)
    transitions, start_chars = _build_bigram_model(tokens)
    lengths = [len(t) for t in tokens]

    if not start_chars:
        return ["" for _ in lengths]

    null_tokens: list[str] = []
    for length in lengths:
        if length == 0:
            null_tokens.append("")
            continue

        start = rng.choice(start_chars)
        chars = [start]

        for _ in range(length - 1):
            prev = chars[-1]
            if prev in transitions:
                chars.append(_sample_char(transitions[prev], rng))
            else:
                chars.append(rng.choice(start_chars))

        null_tokens.append("".join(chars))

    return null_tokens


def generate_null_corpora(
    tokens: list[str],
    n: int = 5,
    base_seed: int = 42,
) -> list[list[str]]:
    """Generate multiple null corpora with different seeds.

    Parameters
    ----------
    tokens : list of str
        Real decoded tokens.
    n : int
        Number of null corpora to generate.
    base_seed : int
        Seeds will be base_seed, base_seed+1, ..., base_seed+n-1.

    Returns
    -------
    list of list of str : n null corpora.
    """
    return [generate_null_corpus(tokens, seed=base_seed + i) for i in range(n)]
