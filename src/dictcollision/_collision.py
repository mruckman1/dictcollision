"""
The collision equation.

    r = sum_{w in D}  prod_{i=1..|w|}  p(w_i)

For every word in the dictionary, multiply together the character
frequencies of the decoded output. Sum. That number is the fraction
of tokens expected to match by chance.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def character_frequencies(tokens: Iterable[str]) -> dict[str, float]:
    """Compute empirical character frequencies over a token stream.

    Parameters
    ----------
    tokens : iterable of str
        Decoded tokens (words).

    Returns
    -------
    dict mapping each character to its relative frequency.
    """
    counts: Counter[str] = Counter()
    for tok in tokens:
        for ch in tok:
            counts[ch] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {c: n / total for c, n in counts.items()}


def token_length_distribution(tokens: Iterable[str]) -> dict[int, int]:
    """Count tokens at each length."""
    dist: dict[int, int] = {}
    for tok in tokens:
        L = len(tok)
        dist[L] = dist.get(L, 0) + 1
    return dist


def noise_floor(
    decoded_tokens: list[str],
    dictionary: Iterable[str],
    char_freqs: dict[str, float] | None = None,
    word_weights: dict[str, float] | None = None,
) -> float:
    """Predict the chance-collision rate for decoded tokens against a dictionary.

    Implements Equation 1 from Ruckman (2026):

        P_hit(L) = sum_{w in D, |w|=L}  prod_i p(w_i)

    aggregated over the token-length distribution:

        r = sum_L  pi(L) * P_hit(L)

    Parameters
    ----------
    decoded_tokens : list of str
        The decoded output stream.
    dictionary : iterable of str
        Dictionary entries to match against.
    char_freqs : dict or None
        Character frequency distribution. If None, computed from
        decoded_tokens.
    word_weights : dict[str, float] or None
        Optional per-word weights applied to chance-collision
        contributions. Default `None` weights every dictionary entry
        equally (the paper's main formulation). Provide a mapping
        (e.g. `-log(corpus_freq)`) when you want the noise floor to
        reflect information value — a high-frequency function word
        contributes its weight times its raw collision probability,
        a rare content morpheme more. Words missing from the dict
        get weight 1.0 (i.e., default behaviour). The result is no
        longer guaranteed to lie in [0, 1] when weights are not
        constant; treat it as a relative quantity in that mode.

    Returns
    -------
    float: predicted fraction of tokens matching by chance. In `[0, 1]`
    when `word_weights is None`; weighted otherwise.

    Examples
    --------
    >>> tokens = ["ab", "cd", "ef", "ab", "gh"]
    >>> dictionary = ["ab", "xy", "cd"]
    >>> noise_floor(tokens, dictionary) > 0
    True
    """
    if not decoded_tokens:
        return 0.0

    if char_freqs is None:
        char_freqs = character_frequencies(decoded_tokens)

    # Group dictionary by length, compute per-word collision probability
    dict_by_len: dict[int, list[str]] = {}
    for w in dictionary:
        dict_by_len.setdefault(len(w), []).append(w)

    weighted = word_weights is not None

    p_hit_by_len: dict[int, float] = {}
    for L, words in dict_by_len.items():
        total_p = 0.0
        for w in words:
            p = 1.0
            for ch in w:
                p *= char_freqs.get(ch, 0.0)
                if p == 0.0:
                    break  # early exit: impossible character
            if weighted:
                total_p += word_weights.get(w, 1.0) * p
            else:
                total_p += p
        p_hit_by_len[L] = total_p if weighted else min(1.0, total_p)

    # Aggregate over decoded corpus length distribution
    len_dist = token_length_distribution(decoded_tokens)
    n_tokens = len(decoded_tokens)
    expected_hits = 0.0
    for L, count_at_L in len_dist.items():
        expected_hits += p_hit_by_len.get(L, 0.0) * count_at_L

    return expected_hits / n_tokens
