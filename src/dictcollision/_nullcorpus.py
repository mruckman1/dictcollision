"""Generate null corpora from character n-gram statistics.

A null corpus preserves the character-pair frequencies and token-length
distribution of the real tokens while destroying word identity. It is
the reference distribution against which the four-category classifier
measures chance collisions.

Three null-model orders are supported:

    - unigram : characters sampled i.i.d. from empirical frequencies
    - bigram  : first-order Markov (default; paper's main model)
    - trigram : second-order Markov (stronger null, used in the paper's
                Section 8 sensitivity analysis)

Stronger nulls preserve more linguistic structure and therefore make
wrong-language evaluations look more clearly negative, at the cost of
reducing the apparent signal on the correct language.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Literal, Sequence

NullModel = Literal["unigram", "bigram", "trigram"]


def _build_bigram_model(
    tokens: Sequence[str],
) -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
    """Build a first-order Markov model from character bigrams."""
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


def _build_trigram_model(
    tokens: Sequence[str],
) -> tuple[
    dict[tuple[str, str], list[tuple[str, float]]],
    dict[str, list[tuple[str, float]]],
    list[str],
]:
    """Build a second-order Markov model."""
    trigram_counts: dict[tuple[str, str], Counter[str]] = {}
    first_pair_counts: dict[str, Counter[str]] = {}
    start_chars: list[str] = []

    for tok in tokens:
        if not tok:
            continue
        start_chars.append(tok[0])
        if len(tok) >= 2:
            if tok[0] not in first_pair_counts:
                first_pair_counts[tok[0]] = Counter()
            first_pair_counts[tok[0]][tok[1]] += 1
        for i in range(len(tok) - 2):
            pair = (tok[i], tok[i + 1])
            if pair not in trigram_counts:
                trigram_counts[pair] = Counter()
            trigram_counts[pair][tok[i + 2]] += 1

    trigram_transitions: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for pair, counts in trigram_counts.items():
        total = sum(counts.values())
        cumulative: list[tuple[str, float]] = []
        running = 0.0
        for c, n in counts.items():
            running += n / total
            cumulative.append((c, running))
        trigram_transitions[pair] = cumulative

    first_pair_transitions: dict[str, list[tuple[str, float]]] = {}
    for c1, counts in first_pair_counts.items():
        total = sum(counts.values())
        cumulative = []
        running = 0.0
        for c2, n in counts.items():
            running += n / total
            cumulative.append((c2, running))
        first_pair_transitions[c1] = cumulative

    return trigram_transitions, first_pair_transitions, start_chars


def _sample_char(table: list[tuple[str, float]], rng: random.Random) -> str:
    r = rng.random()
    for char, cum_prob in table:
        if r <= cum_prob:
            return char
    return table[-1][0]


def _unigram_table(tokens: Sequence[str]) -> tuple[list[tuple[str, float]], list[str]]:
    counts: Counter[str] = Counter()
    starts: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        starts.append(tok[0])
        for ch in tok:
            counts[ch] += 1
    total = sum(counts.values())
    if total == 0:
        return [], starts
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for c, n in counts.items():
        running += n / total
        cumulative.append((c, running))
    return cumulative, starts


def generate_null_corpus(
    tokens: list[str],
    seed: int = 42,
    null_model: NullModel = "bigram",
) -> list[str]:
    """Generate one null corpus.

    Parameters
    ----------
    tokens : list of str
        Real decoded tokens to mimic.
    seed : int
        Random seed for reproducibility.
    null_model : {"unigram", "bigram", "trigram"}
        Order of the n-gram model used for sampling.

    Returns
    -------
    list of str
        Null tokens with same length distribution, character n-gram
        frequencies preserved, word identity destroyed.
    """
    if not tokens:
        return []

    rng = random.Random(seed)
    lengths = [len(t) for t in tokens]

    if null_model == "unigram":
        table, starts = _unigram_table(tokens)
        if not table:
            return ["" for _ in lengths]
        out: list[str] = []
        for length in lengths:
            if length == 0:
                out.append("")
                continue
            out.append("".join(_sample_char(table, rng) for _ in range(length)))
        return out

    if null_model == "bigram":
        transitions, start_chars = _build_bigram_model(tokens)
        if not start_chars:
            return ["" for _ in lengths]
        out = []
        for length in lengths:
            if length == 0:
                out.append("")
                continue
            chars = [rng.choice(start_chars)]
            for _ in range(length - 1):
                prev = chars[-1]
                if prev in transitions:
                    chars.append(_sample_char(transitions[prev], rng))
                else:
                    chars.append(rng.choice(start_chars))
            out.append("".join(chars))
        return out

    if null_model == "trigram":
        trigram_t, first_pair_t, start_chars = _build_trigram_model(tokens)
        bigram_t, _ = _build_bigram_model(tokens)  # fallback if trigram miss
        if not start_chars:
            return ["" for _ in lengths]
        out = []
        for length in lengths:
            if length == 0:
                out.append("")
                continue
            chars = [rng.choice(start_chars)]
            if length >= 2:
                c1 = chars[0]
                if c1 in first_pair_t:
                    chars.append(_sample_char(first_pair_t[c1], rng))
                elif c1 in bigram_t:
                    chars.append(_sample_char(bigram_t[c1], rng))
                else:
                    chars.append(rng.choice(start_chars))
            for _ in range(length - 2):
                pair = (chars[-2], chars[-1])
                if pair in trigram_t:
                    chars.append(_sample_char(trigram_t[pair], rng))
                elif chars[-1] in bigram_t:
                    chars.append(_sample_char(bigram_t[chars[-1]], rng))
                else:
                    chars.append(rng.choice(start_chars))
            out.append("".join(chars))
        return out

    raise ValueError(
        f"null_model must be 'unigram', 'bigram', or 'trigram', got {null_model!r}"
    )


def generate_null_corpora(
    tokens: list[str],
    n: int = 5,
    base_seed: int = 42,
    null_model: NullModel = "bigram",
) -> list[list[str]]:
    """Generate multiple null corpora with different seeds.

    Parameters
    ----------
    tokens : list of str
    n : int
        Number of null corpora.
    base_seed : int
        Seeds will be base_seed, base_seed+1, ..., base_seed+n-1.
    null_model : {"unigram", "bigram", "trigram"}
    """
    return [
        generate_null_corpus(tokens, seed=base_seed + i, null_model=null_model)
        for i in range(n)
    ]
