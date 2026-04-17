"""Four-category token classification framework."""

from __future__ import annotations

from collections import Counter

from dictcollision._nullcorpus import generate_null_corpora
from dictcollision._types import ClassifyResult


def classify(
    decoded_tokens: list[str],
    dictionary: set[str] | list[str],
    n_nulls: int = 5,
    threshold: int = 2,
    base_seed: int = 42,
    null_corpora: list[list[str]] | None = None,
) -> ClassifyResult:
    """Classify every decoded token type into one of four categories.

    Categories
    ----------
    - **signal**: hits dictionary in real text, appears in fewer than
      `threshold` null corpora.
    - **shared_hit**: hits in both real and at least `threshold` nulls
      (chance collision on a real word).
    - **anti_signal**: hits in at least `threshold` nulls but not in real
      (phantom match).
    - **shared_miss**: misses in both, or not in dictionary.

    Parameters
    ----------
    decoded_tokens : list of str
        Decoded output to evaluate.
    dictionary : set or list of str
        Reference dictionary.
    n_nulls : int
        Number of null corpora to generate (default 5).
    threshold : int
        A word type counts as "appearing in nulls" if it appears in
        >= threshold of the n_nulls corpora (default 2).
    base_seed : int
        Seed for null corpus generation.
    null_corpora : list of list of str, optional
        Pre-generated null corpora. If provided, n_nulls and base_seed
        are ignored.

    Returns
    -------
    ClassifyResult
    """
    dict_set = dictionary if isinstance(dictionary, set) else set(dictionary)

    if null_corpora is None:
        nulls = generate_null_corpora(decoded_tokens, n=n_nulls, base_seed=base_seed)
    else:
        nulls = null_corpora
        n_nulls = len(nulls)

    n_tokens = len(decoded_tokens)
    if n_tokens == 0:
        return ClassifyResult(
            signal=0.0,
            shared_hit=0.0,
            anti_signal=0.0,
            shared_miss=1.0,
            net_signal=0.0,
            apparent_hit_rate=0.0,
            correction=0.0,
            n_tokens=0,
        )

    real_counts = Counter(decoded_tokens)

    # Count how many nulls each type appears in, plus sum of counts
    null_presence: Counter[str] = Counter()
    null_counts_sum: dict[str, float] = {}
    for null_corpus in nulls:
        seen: set[str] = set()
        for tok in null_corpus:
            null_counts_sum[tok] = null_counts_sum.get(tok, 0.0) + 1
            seen.add(tok)
        for w in seen:
            null_presence[w] += 1

    effective_nulls = max(1, n_nulls)
    null_mean: dict[str, float] = {
        w: null_counts_sum[w] / effective_nulls for w in null_counts_sum
    }

    signal_tokens = 0
    shared_hit_tokens = 0
    anti_signal_tokens = 0.0
    shared_miss_tokens = 0
    signal_words: list[str] = []
    anti_signal_words: list[str] = []

    all_types: set[str] = set(real_counts.keys())
    for null_corpus in nulls:
        all_types.update(null_corpus)

    for w in all_types:
        in_dict = w in dict_set
        in_real = w in real_counts
        in_null = null_presence.get(w, 0) >= threshold

        if in_dict and in_real and not in_null:
            signal_tokens += real_counts[w]
            signal_words.append(w)
        elif in_dict and in_real and in_null:
            shared_hit_tokens += real_counts[w]
        elif in_dict and (not in_real) and in_null:
            anti_signal_tokens += null_mean.get(w, 0.0)
            anti_signal_words.append(w)
        else:
            if in_real:
                shared_miss_tokens += real_counts[w]

    signal_frac = signal_tokens / n_tokens
    shared_hit_frac = shared_hit_tokens / n_tokens
    anti_signal_frac = anti_signal_tokens / n_tokens
    shared_miss_frac = shared_miss_tokens / n_tokens

    net = signal_frac - anti_signal_frac
    apparent = signal_frac + shared_hit_frac

    signal_words.sort(key=lambda w: real_counts.get(w, 0), reverse=True)

    return ClassifyResult(
        signal=signal_frac,
        shared_hit=shared_hit_frac,
        anti_signal=anti_signal_frac,
        shared_miss=shared_miss_frac,
        net_signal=net,
        apparent_hit_rate=apparent,
        correction=apparent - net,
        n_tokens=n_tokens,
        signal_words=signal_words,
        anti_signal_words=sorted(anti_signal_words),
    )
