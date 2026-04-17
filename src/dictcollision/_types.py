"""Shared types for dictcollision."""

from __future__ import annotations

from dataclasses import dataclass, field


def _bar(fraction: float, width: int = 23) -> str:
    """Render a text bar for `fraction in [0, 1]` using unicode blocks."""
    if fraction < 0:
        fraction = 0.0
    if fraction > 1:
        fraction = 1.0
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def _interpret_net_signal(net: float) -> str:
    if net >= 0.20:
        return "strong signal — dictionary is a good fit"
    if net >= 0.05:
        return "partial signal — possibly correct, check anti_signal_words"
    if net > -0.02:
        return "no signal beyond chance"
    return "worse than random — wrong language or decode table"


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

    def summary(self) -> str:
        """Human-readable interpretation of the result.

        Returns a multi-line string suitable for printing. See the
        README section "Interpreting net_signal" for the thresholds.
        """
        lines = [
            f"ClassifyResult (n={self.n_tokens} tokens)",
            f"  apparent hit rate : {self.apparent_hit_rate:>6.1%}",
            f"  net signal        : {self.net_signal:>6.1%}   <- calibrated metric",
            f"  correction        : {self.correction:>6.1%}   <- amount subtracted",
            "",
            f"  signal       {self.signal:>5.1%}  {_bar(self.signal)}  real matches",
            f"  shared_hit   {self.shared_hit:>5.1%}  {_bar(self.shared_hit)}  chance collisions",
            f"  anti_signal  {self.anti_signal:>5.1%}  {_bar(self.anti_signal)}  phantom matches",
            f"  shared_miss  {self.shared_miss:>5.1%}  {_bar(self.shared_miss)}  non-dict tokens",
            "",
            f"  Interpretation: {_interpret_net_signal(self.net_signal)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Plain-dict form suitable for JSON serialization."""
        return {
            "n_tokens": self.n_tokens,
            "signal": self.signal,
            "shared_hit": self.shared_hit,
            "anti_signal": self.anti_signal,
            "shared_miss": self.shared_miss,
            "net_signal": self.net_signal,
            "apparent_hit_rate": self.apparent_hit_rate,
            "correction": self.correction,
            "n_signal_words": len(self.signal_words),
            "n_anti_signal_words": len(self.anti_signal_words),
            "signal_words_top20": self.signal_words[:20],
            "anti_signal_words_top20": self.anti_signal_words[:20],
        }


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

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.name:<24}  observed={self.observed_hit_rate:>6.1%}  "
            f"predicted={self.predicted_noise:>6.1%}  "
            f"excess={self.excess:>6.1%}  snr={self.snr:>5.1f}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "predicted_noise": self.predicted_noise,
            "observed_hit_rate": self.observed_hit_rate,
            "excess": self.excess,
            "snr": self.snr,
            "n_tokens": self.n_tokens,
            "n_hits": self.n_hits,
        }


@dataclass(frozen=True)
class LengthBucket:
    """Four-category breakdown for a single decoded-token length."""

    length: int
    n_tokens: int
    apparent_hit_rate: float
    net_signal: float
    correction: float
    signal: float
    shared_hit: float
    anti_signal: float
    shared_miss: float

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "n_tokens": self.n_tokens,
            "apparent_hit_rate": self.apparent_hit_rate,
            "net_signal": self.net_signal,
            "correction": self.correction,
            "signal": self.signal,
            "shared_hit": self.shared_hit,
            "anti_signal": self.anti_signal,
            "shared_miss": self.shared_miss,
        }


@dataclass(frozen=True)
class NullDistribution:
    """Empirical distribution of net_signal under the null hypothesis of
    no semantic signal, generated by bigram-resampling the token stream.

    Use `percentile_of(observed)` to report where an observed value falls
    in this distribution.
    """

    net_signals: list[float]
    observed_net_signal: float
    n_samples: int

    @property
    def median(self) -> float:
        s = sorted(self.net_signals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    @property
    def mean(self) -> float:
        if not self.net_signals:
            return 0.0
        return sum(self.net_signals) / len(self.net_signals)

    def percentile_of(self, value: float) -> float:
        """Percentile (0-100) of `value` within the null distribution."""
        if not self.net_signals:
            return 50.0
        below = sum(1 for v in self.net_signals if v < value)
        return 100.0 * below / len(self.net_signals)

    def observed_percentile(self) -> float:
        return self.percentile_of(self.observed_net_signal)

    def summary(self) -> str:
        return (
            f"NullDistribution (n={self.n_samples} Monte Carlo samples)\n"
            f"  null median   : {self.median:>6.1%}\n"
            f"  null mean     : {self.mean:>6.1%}\n"
            f"  observed      : {self.observed_net_signal:>6.1%}\n"
            f"  observed pct  : {self.observed_percentile():>5.1f}%  "
            "(100 = clearly above null)"
        )


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for net_signal."""

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_samples: int

    def summary(self) -> str:
        pct = int(round(self.confidence * 100))
        return (
            f"net_signal = {self.point_estimate:.1%} "
            f"[{pct}% CI: {self.lower:.1%}, {self.upper:.1%}]"
        )
