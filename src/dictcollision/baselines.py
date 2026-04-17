"""Standard correction baselines for the dictionary collision problem.

Each function in this module returns a `signal fraction of real tokens` that
can be compared head-to-head with `dictcollision.classify(...).net_signal`.
These reproduce the five alternatives tested in Table 2 and Figure 5 of
Ruckman (2026):

    - apparent_hit_rate       : no correction
    - subtract_null           : subtract the mean null apparent rate
    - permutation_test        : per-word Poisson tail test, aggregated
    - bh_fdr                  : permutation p-values with Benjamini-Hochberg
    - blast_evalue            : per-word BLAST-style E-value acceptance

Each returns the fraction of real tokens whose types the method counts as
"signal," so higher = the method is more optimistic about the hit being real.

The paper's four-category `net_signal` is the only method that correctly
flags wrong-language evaluations as worse than chance; these baselines are
provided so users can reproduce that head-to-head comparison on their own
data.
"""

from __future__ import annotations

import math
from collections import Counter

from dictcollision._nullcorpus import generate_null_corpora


def _apparent_rate(tokens: list[str], dictionary: set[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in dictionary) / len(tokens)


def apparent_hit_rate(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
) -> float:
    """No correction. The raw fraction of tokens in the dictionary."""
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)
    return _apparent_rate(decoded_tokens, dict_set)


def subtract_null(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    n_nulls: int = 5,
    base_seed: int = 42,
    null_corpora: list[list[str]] | None = None,
) -> float:
    """Apparent hit rate minus the mean apparent rate across null corpora.

    This is the simplest correction: compute a null baseline by running the
    same dictionary lookup on bigram-resampled tokens, then subtract. Paper
    Section 6 shows this over-attributes signal in the partial-decipherment
    regime, typically by ~12 percentage points at 200K.
    """
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)
    nulls = null_corpora or generate_null_corpora(
        decoded_tokens, n=n_nulls, base_seed=base_seed
    )
    observed = _apparent_rate(decoded_tokens, dict_set)
    null_rates = [_apparent_rate(n, dict_set) for n in nulls]
    mean_null = sum(null_rates) / len(null_rates) if null_rates else 0.0
    return observed - mean_null


def _poisson_sf(k: int, mu: float) -> float:
    """Survival function P(X >= k) for X ~ Poisson(mu), via simple summation.

    Robust to mu=0 and reasonable for small-to-moderate k (<~500), which is
    the regime we actually hit with word counts.
    """
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    # cdf = P(X <= k-1); survival = 1 - cdf
    # Compute pmf iteratively.
    cdf = 0.0
    term = math.exp(-mu)  # pmf at 0
    cdf += term
    for i in range(1, k):
        term *= mu / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def _per_word_stats(
    decoded_tokens: list[str],
    nulls: list[list[str]],
    dict_set: set[str],
) -> dict[str, tuple[int, float]]:
    """For each dict-matching type, return (real_count, mean_null_count)."""
    real = Counter(decoded_tokens)
    n_nulls = max(1, len(nulls))
    null_counts_sum: dict[str, float] = {}
    for nc in nulls:
        c = Counter(nc)
        for w, cnt in c.items():
            null_counts_sum[w] = null_counts_sum.get(w, 0.0) + cnt

    stats: dict[str, tuple[int, float]] = {}
    all_types = set(real.keys())
    for nc in nulls:
        all_types.update(nc)
    for w in all_types:
        if w not in dict_set:
            continue
        r = real.get(w, 0)
        mu = null_counts_sum.get(w, 0.0) / n_nulls
        stats[w] = (r, mu)
    return stats


def permutation_test(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    alpha: float = 0.05,
    n_nulls: int = 5,
    base_seed: int = 42,
    null_corpora: list[list[str]] | None = None,
) -> float:
    """Per-word Poisson tail test, aggregated to token-fraction signal.

    For each dict-matching type w with real count r and mean null count mu,
    compute p = P(X >= r | X ~ Poisson(mu)). A type with p < alpha is
    accepted. Signal = fraction of real tokens whose type was accepted.

    Paper Section 6.1 shows this fails at wrong-language evaluations:
    rare dict-matches appear in real but never in small null samples, so
    p tends to zero by construction and the word is accepted even though
    it's chance.
    """
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)
    if not decoded_tokens:
        return 0.0
    nulls = null_corpora or generate_null_corpora(
        decoded_tokens, n=n_nulls, base_seed=base_seed
    )
    stats = _per_word_stats(decoded_tokens, nulls, dict_set)
    real = Counter(decoded_tokens)

    accepted_tokens = 0
    for w, (r, mu) in stats.items():
        if r == 0:
            continue
        p = _poisson_sf(r, mu)
        if p < alpha:
            accepted_tokens += real.get(w, 0)
    return accepted_tokens / len(decoded_tokens)


def bh_fdr(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    q: float = 0.05,
    n_nulls: int = 5,
    base_seed: int = 42,
    null_corpora: list[list[str]] | None = None,
) -> float:
    """Permutation p-values with Benjamini-Hochberg FDR control.

    Same per-word p-values as permutation_test, controlled at FDR q via BH.
    Signal = fraction of real tokens whose type survives BH.
    """
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)
    if not decoded_tokens:
        return 0.0
    nulls = null_corpora or generate_null_corpora(
        decoded_tokens, n=n_nulls, base_seed=base_seed
    )
    stats = _per_word_stats(decoded_tokens, nulls, dict_set)
    real = Counter(decoded_tokens)

    pvals: list[tuple[str, float]] = []
    for w, (r, mu) in stats.items():
        if r == 0:
            continue
        pvals.append((w, _poisson_sf(r, mu)))

    if not pvals:
        return 0.0

    pvals.sort(key=lambda x: x[1])
    m = len(pvals)
    # Find largest k such that p_(k) <= k/m * q
    accepted: set[str] = set()
    for k, (w, p) in enumerate(pvals, start=1):
        if p <= k / m * q:
            accepted.add(w)

    accepted_tokens = sum(real.get(w, 0) for w in accepted)
    return accepted_tokens / len(decoded_tokens)


def blast_evalue(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    threshold: float = 1.0,
    n_nulls: int = 5,
    base_seed: int = 42,
    null_corpora: list[list[str]] | None = None,
) -> float:
    """BLAST-style E-value acceptance.

    For each dict-matching type of length L with real count r, compute
    E = n_dict_L * P(X >= r | X ~ Poisson(mu)), where mu is the mean
    null count for that type and n_dict_L is the number of dictionary
    entries of length L (the search-space correction).

    A type is accepted if E < threshold (default 1). Signal = fraction
    of real tokens in accepted types. Paper Section 6 shows this shares
    the single-word-test weakness: phantom matches absent from nulls
    are not detected.
    """
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)
    if not decoded_tokens:
        return 0.0
    nulls = null_corpora or generate_null_corpora(
        decoded_tokens, n=n_nulls, base_seed=base_seed
    )
    stats = _per_word_stats(decoded_tokens, nulls, dict_set)
    real = Counter(decoded_tokens)

    dict_length_counts: Counter[int] = Counter(len(w) for w in dict_set)

    accepted_tokens = 0
    for w, (r, mu) in stats.items():
        if r == 0:
            continue
        p = _poisson_sf(r, mu)
        e = dict_length_counts[len(w)] * p
        if e < threshold:
            accepted_tokens += real.get(w, 0)
    return accepted_tokens / len(decoded_tokens)


def all_methods(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    n_nulls: int = 5,
    base_seed: int = 42,
) -> dict[str, float]:
    """Run all baselines + framework's net signal in one pass.

    Shares null corpora across methods for consistency and speed. Returns
    a dict suitable for reproducing Table 2 from the paper.
    """
    from dictcollision._framework import classify

    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)
    nulls = generate_null_corpora(decoded_tokens, n=n_nulls, base_seed=base_seed)

    result = classify(decoded_tokens, dict_set, null_corpora=nulls)
    return {
        "apparent_hit_rate": apparent_hit_rate(decoded_tokens, dict_set),
        "subtract_null": subtract_null(decoded_tokens, dict_set, null_corpora=nulls),
        "permutation_test": permutation_test(
            decoded_tokens, dict_set, null_corpora=nulls
        ),
        "bh_fdr": bh_fdr(decoded_tokens, dict_set, null_corpora=nulls),
        "blast_evalue": blast_evalue(
            decoded_tokens, dict_set, null_corpora=nulls
        ),
        "four_category_net": result.net_signal,
    }
