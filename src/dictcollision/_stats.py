"""Monte Carlo null distribution and bootstrap confidence intervals."""

from __future__ import annotations

import random

from dictcollision._framework import classify
from dictcollision._nullcorpus import NullModel, generate_null_corpus
from dictcollision._types import BootstrapCI, NullDistribution


def null_distribution(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    n: int = 40,
    n_nulls_per_sample: int = 5,
    base_seed: int = 42,
    null_model: NullModel = "bigram",
) -> NullDistribution:
    """Empirical null distribution of `net_signal`.

    For `n` iterations, generate a bigram-resampled "pseudo-real" corpus
    from the original tokens, classify it against the dictionary using
    fresh null corpora, and collect the resulting net_signal. This
    approximates the distribution of net_signal under the null hypothesis
    of no semantic signal (the bigram-resampled corpus has matched
    character statistics but no real linguistic content).

    Paper Section 8.1 uses a similar random-table-decoding Monte Carlo;
    this version works directly from decoded tokens (no cipher needed).

    Parameters
    ----------
    decoded_tokens : list of str
        The original real corpus.
    dictionary : set or list of str
        Reference dictionary.
    n : int
        Number of Monte Carlo samples (default 40).
    n_nulls_per_sample : int
        Null corpora per classify call (default 5).
    base_seed : int
        Seed for reproducibility.
    null_model : {"unigram", "bigram", "trigram"}

    Returns
    -------
    NullDistribution
        Holds the observed net_signal on the real corpus alongside the
        Monte Carlo distribution so users can call `observed_percentile()`.
    """
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)

    observed = classify(
        decoded_tokens,
        dict_set,
        n_nulls=n_nulls_per_sample,
        base_seed=base_seed,
        null_model=null_model,
    ).net_signal

    nets: list[float] = []
    for i in range(n):
        pseudo_real = generate_null_corpus(
            decoded_tokens,
            seed=base_seed + 10_000 + i,
            null_model=null_model,
        )
        r = classify(
            pseudo_real,
            dict_set,
            n_nulls=n_nulls_per_sample,
            base_seed=base_seed + 20_000 + i * n_nulls_per_sample,
            null_model=null_model,
        )
        nets.append(r.net_signal)

    return NullDistribution(
        net_signals=nets,
        observed_net_signal=observed,
        n_samples=n,
    )


def bootstrap_ci(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    n: int = 200,
    confidence: float = 0.95,
    n_nulls_per_sample: int = 5,
    base_seed: int = 42,
    null_model: NullModel = "bigram",
) -> BootstrapCI:
    """Bootstrap confidence interval on `net_signal`.

    Resamples the decoded token stream with replacement `n` times, re-runs
    `classify` on each resample, and returns the percentile-based CI at
    the requested confidence level.

    Parameters
    ----------
    n : int
        Number of bootstrap resamples (default 200). Increase for
        tighter CIs at the cost of runtime — each resample runs a
        full classify pass.
    confidence : float
        Two-sided confidence level (default 0.95).
    """
    if not decoded_tokens:
        return BootstrapCI(0.0, 0.0, 0.0, confidence, 0)

    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)

    point = classify(
        decoded_tokens,
        dict_set,
        n_nulls=n_nulls_per_sample,
        base_seed=base_seed,
        null_model=null_model,
    ).net_signal

    rng = random.Random(base_seed)
    m = len(decoded_tokens)
    nets: list[float] = []
    for i in range(n):
        resample = [decoded_tokens[rng.randrange(m)] for _ in range(m)]
        r = classify(
            resample,
            dict_set,
            n_nulls=n_nulls_per_sample,
            base_seed=base_seed + 30_000 + i,
            null_model=null_model,
        )
        nets.append(r.net_signal)

    nets.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n)
    hi_idx = int((1.0 - alpha) * n) - 1
    lo_idx = max(0, min(n - 1, lo_idx))
    hi_idx = max(0, min(n - 1, hi_idx))

    return BootstrapCI(
        point_estimate=point,
        lower=nets[lo_idx],
        upper=nets[hi_idx],
        confidence=confidence,
        n_samples=n,
    )
