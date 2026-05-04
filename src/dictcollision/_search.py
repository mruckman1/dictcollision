"""Search-procedure calibration for stochastic decipherment.

When the decoded tokens fed to `classify` are themselves the output of
a stochastic search over a key space (e.g. simulated annealing on a
substitution alphabet), the search itself can manufacture apparent
signal: a quadgram-optimised key on a short cipher will find local
optima that resolve into a handful of high-frequency dictionary words
even when the cipher has no underlying linguistic structure.

`search_calibrated_signal` answers: does the search find more signal
on the real cipher than on random shuffles of the cipher's symbol
multiset? The shuffles preserve the alphabet-search budget and the
character-multiset constraint while destroying any positional
linguistic content. A high z-score of the real observation against
the shuffle distribution is the calibrated signal.

This is the right primitive for the case the existing
`null_distribution` does not cover. `null_distribution` holds the
decode fixed and asks whether a *fixed* decoded token stream's signal
is distinguishable from a bigram-resampled null. `search_calibrated_signal`
asks whether the *search procedure* finds more signal on a real cipher
than on a permutation of it. Both are useful; reach for this one when
the decoded tokens came from a key-space search.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

from dictcollision._framework import classify
from dictcollision._nullcorpus import NullModel
from dictcollision._types import SearchCalibrationResult


def search_calibrated_signal(
    cipher_symbols: Sequence[Any],
    search_fn: Callable[[list], list[str]],
    dictionary: set[str] | list[str],
    n_shuffles: int = 30,
    base_seed: int = 42,
    null_model: NullModel = "bigram",
    n_nulls: int = 5,
    threshold: int = 2,
) -> SearchCalibrationResult:
    """Net signal calibrated against a matched-budget shuffle baseline.

    Runs `search_fn` on the cipher and on `n_shuffles` random shuffles
    of the cipher's symbol multiset. Each gets its own independent
    search (matched budget). Returns net_signal on the real cipher
    along with a z-score and percentile against the shuffle
    distribution of net_signals.

    Use this when your decoded tokens are produced by a stochastic
    search over a key space (SA, hill-climbing, AZdecrypt, ...). The
    raw `net_signal` of any single decode can be misleading at short
    lengths because the search can find chance local optima with
    matched character statistics; the shuffle baseline corrects for
    that.

    Parameters
    ----------
    cipher_symbols : sequence
        The cipher itself, as a sequence of any hashable symbols
        (characters, ints, tuples). Not the decoded tokens — those are
        produced by `search_fn`.
    search_fn : callable taking a list of cipher symbols, returning a
        list of decoded token strings
        Should be deterministic given a fixed cipher *or* internally
        seeded; this function does not pass an RNG to `search_fn`.
        Re-running `search_fn` on the same input must produce
        comparable output for the calibration to be meaningful — if
        your search is itself randomised, seed it on the input or on a
        fixed seed before calling.
    dictionary : set or list of str
    n_shuffles : int
        Number of random multiset-preserving shuffles to compare
        against (default 30).
    base_seed : int
        Seeds the shuffles and the per-call classify nulls.
    null_model : {"unigram", "bigram", "trigram"}
        Passed through to each `classify` call when scoring decoded
        tokens.
    n_nulls : int
        Null corpora per `classify` call (default 5).
    threshold : int
        Min nulls a word must appear in to be 'shared' (default 2).

    Returns
    -------
    SearchCalibrationResult

    Raises
    ------
    ValueError
        If `cipher_symbols` is empty or `n_shuffles < 1`.

    Notes
    -----
    Determinism: every shuffle uses a derived seed of the form
    `base_seed + shuffle_index + 1`; every classify call uses a derived
    seed `base_seed + 30000 + shuffle_index * n_nulls`. With a fixed
    `search_fn` and `base_seed`, results are reproducible.
    """
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be >= 1, got {n_shuffles}")

    cipher_list = list(cipher_symbols)
    if not cipher_list:
        raise ValueError("cipher_symbols must be non-empty")

    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)

    observed_tokens = search_fn(list(cipher_list))
    observed_net = classify(
        list(observed_tokens),
        dict_set,
        n_nulls=n_nulls,
        threshold=threshold,
        base_seed=base_seed,
        null_model=null_model,
    ).net_signal

    shuffle_nets: list[float] = []
    for i in range(n_shuffles):
        shuffled = list(cipher_list)
        rng = random.Random(base_seed + i + 1)
        rng.shuffle(shuffled)
        decoded = search_fn(shuffled)
        r = classify(
            list(decoded),
            dict_set,
            n_nulls=n_nulls,
            threshold=threshold,
            base_seed=base_seed + 30_000 + i * max(1, n_nulls),
            null_model=null_model,
        )
        shuffle_nets.append(r.net_signal)

    if n_shuffles >= 1:
        mean = sum(shuffle_nets) / n_shuffles
    else:
        mean = 0.0

    if n_shuffles >= 2:
        var = sum((x - mean) ** 2 for x in shuffle_nets) / (n_shuffles - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    if std == 0.0:
        z = 0.0
    else:
        z = (observed_net - mean) / std

    below = sum(1 for v in shuffle_nets if v < observed_net)
    percentile = 100.0 * below / n_shuffles if n_shuffles else 50.0

    return SearchCalibrationResult(
        observed_net_signal=observed_net,
        shuffle_net_signals=shuffle_nets,
        shuffle_mean=mean,
        shuffle_std=std,
        z_score=z,
        percentile=percentile,
        n_shuffles=n_shuffles,
        n_cipher_symbols=len(cipher_list),
    )
