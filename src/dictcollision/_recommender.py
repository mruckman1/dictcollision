"""Dictionary recommender: rank dictionaries by signal quality."""

from __future__ import annotations

from typing import Literal

from dictcollision._collision import character_frequencies, noise_floor
from dictcollision._types import Recommendation


def recommend(
    decoded_tokens: list[str],
    dictionaries: dict[str, set[str] | list[str]],
    objective: Literal["excess", "snr"] = "excess",
) -> list[Recommendation]:
    """Rank candidate dictionaries by signal quality.

    Parameters
    ----------
    decoded_tokens : list of str
        Decoded output to evaluate.
    dictionaries : dict mapping name -> word set
        Candidate dictionaries to rank.
    objective : "excess" or "snr"
        - "excess": maximize (observed - predicted_noise).
          Picks the dictionary with the largest absolute signal.
        - "snr": maximize observed / predicted_noise.
          Picks the dictionary where signal is most confident.

    Returns
    -------
    list of Recommendation, sorted best-first.

    Examples
    --------
    >>> tokens = ["the", "cat", "the"]
    >>> results = recommend(tokens, {"en": {"the", "cat"}, "de": {"und"}})
    >>> results[0].name
    'en'
    """
    if objective not in {"excess", "snr"}:
        raise ValueError(f"objective must be 'excess' or 'snr', got {objective!r}")

    if not decoded_tokens:
        return []

    char_freqs = character_frequencies(decoded_tokens)
    n_tokens = len(decoded_tokens)
    results: list[Recommendation] = []

    for name, words in dictionaries.items():
        word_set = words if isinstance(words, set) else set(words)
        predicted = noise_floor(decoded_tokens, word_set, char_freqs)
        n_hits = sum(1 for t in decoded_tokens if t in word_set)
        observed = n_hits / n_tokens
        excess = max(0.0, observed - predicted)

        if n_hits == 0:
            snr = 0.0
        else:
            p_floor = max(predicted, 0.5 / n_tokens)
            snr = observed / p_floor

        results.append(
            Recommendation(
                name=name,
                predicted_noise=predicted,
                observed_hit_rate=observed,
                excess=excess,
                snr=snr,
                n_tokens=n_tokens,
                n_hits=n_hits,
            )
        )

    results.sort(
        key=lambda r: r.excess if objective == "excess" else r.snr,
        reverse=True,
    )
    return results
