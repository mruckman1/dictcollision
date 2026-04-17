"""Shared types for dictcollision."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassifyResult:
    """Result of the four-category token classification.

    Attributes
    ----------
    signal : float
        Fraction of tokens that hit the dictionary in real text
        but not in null corpora. Genuine matches.
    shared_hit : float
        Fraction hitting in both real and null. Chance collisions
        that happen to also be real words.
    anti_signal : float
        Fraction hitting in null but not real. Phantom matches
        that inflate any naive baseline subtraction.
    shared_miss : float
        Fraction missing in both. Non-dictionary tokens.
    net_signal : float
        signal - anti_signal. The calibrated metric.
    apparent_hit_rate : float
        signal + shared_hit. What a naive evaluator would report.
    correction : float
        apparent_hit_rate - net_signal. How much the framework corrects.
    n_tokens : int
        Total token count in the decoded corpus.
    signal_words : list[str]
        Word types classified as signal.
    anti_signal_words : list[str]
        Word types classified as anti-signal.
    """

    signal: float
    shared_hit: float
    anti_signal: float
    shared_miss: float
    net_signal: float
    apparent_hit_rate: float
    correction: float
    n_tokens: int
    signal_words: list[str] = field(default_factory=list)
    anti_signal_words: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Recommendation:
    """Ranking result for a single candidate dictionary."""

    name: str
    predicted_noise: float
    observed_hit_rate: float
    excess: float
    snr: float
    n_tokens: int
    n_hits: int
