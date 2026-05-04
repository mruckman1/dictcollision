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


def _interpret_net_signal(net: float, n_tokens: int | None = None) -> str:
    if net >= 0.20:
        base = "strong signal — dictionary is a good fit"
    elif net >= 0.05:
        base = "partial signal — possibly correct, check anti_signal_words"
    elif net > -0.02:
        base = "no signal beyond chance"
    else:
        base = "worse than random — wrong language or decode table"

    if n_tokens is not None and n_tokens < 200:
        suffix = (
            f"\n  Note: text is short (n={n_tokens}). If decode came from "
            "stochastic key search,\n  run search_calibrated_signal() — "
            "absolute net_signal can mislead in this regime."
        )
        return base + suffix
    return base


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
        Word types classified as signal, sorted by real-corpus count desc.
    anti_signal_words : list[str]
        Word types classified as anti-signal.
    signal_word_counts : dict[str, int]
        Per-word real-corpus counts for signal types. Sums to
        signal * n_tokens.
    anti_signal_word_counts : dict[str, float]
        Per-word mean null-corpus counts for anti-signal types (floats
        because they are means across n_nulls corpora). Sum approximates
        anti_signal * n_tokens.
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
    signal_word_counts: dict[str, int] = field(default_factory=dict)
    anti_signal_word_counts: dict[str, float] = field(default_factory=dict)

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
            f"  Interpretation: {_interpret_net_signal(self.net_signal, self.n_tokens)}",
        ]
        return "\n".join(lines)

    def overfit_score(self) -> float:
        """Heuristic concentration score for the signal token mass.

        Returns the fraction of signal tokens accounted for by the top
        three signal word types. Real text typically sits below ~0.4;
        SA-overfit decodes (e.g., a quadgram-optimised key that resolves
        the cipher into a few repeated dictionary words) often exceed
        ~0.7.

        Thresholds are heuristic and require empirical calibration on
        your domain. Returns 0.0 when there are no signal words.
        """
        if not self.signal_word_counts:
            return 0.0
        counts = sorted(self.signal_word_counts.values(), reverse=True)
        total = sum(counts)
        if total == 0:
            return 0.0
        top3 = sum(counts[:3])
        return top3 / total

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
            "signal_word_counts_top20": [
                [w, self.signal_word_counts.get(w, 0)]
                for w in self.signal_words[:20]
            ],
            "anti_signal_word_counts_top20": [
                [w, self.anti_signal_word_counts.get(w, 0.0)]
                for w in self.anti_signal_words[:20]
            ],
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


@dataclass(frozen=True)
class SearchCalibrationResult:
    """Matched-budget shuffle calibration of a stochastic decode search.

    Reports the net_signal of the search applied to the real cipher
    alongside the distribution of net_signals obtained when the same
    search procedure is applied to random shuffles of the cipher's
    symbol multiset. The shuffle distribution captures what the search
    can find by chance with the same character-multiset budget; the
    z-score of the observed value against that distribution is the
    calibrated signal.

    See `dictcollision.search_calibrated_signal` for the constructor.

    Attributes
    ----------
    observed_net_signal : float
        net_signal from running search_fn on the real cipher.
    shuffle_net_signals : list[float]
        net_signal per shuffled-cipher run (length = n_shuffles).
    shuffle_mean, shuffle_std : float
        Sample mean and (n-1) standard deviation of shuffle_net_signals.
        shuffle_std is 0.0 when n_shuffles < 2 or all shuffles tied.
    z_score : float
        (observed - shuffle_mean) / shuffle_std. 0.0 when shuffle_std == 0.
    percentile : float
        Percentile (0-100) of observed_net_signal within the shuffle
        distribution. 100 = strictly above all shuffles.
    n_shuffles : int
    n_cipher_symbols : int
    """

    observed_net_signal: float
    shuffle_net_signals: list[float]
    shuffle_mean: float
    shuffle_std: float
    z_score: float
    percentile: float
    n_shuffles: int
    n_cipher_symbols: int

    def _interpretation(self) -> str:
        if self.shuffle_std == 0.0:
            return (
                "shuffle distribution is degenerate (zero variance) — "
                "search may be deterministic w.r.t. the symbol multiset"
            )
        if self.z_score >= 3.0:
            return "above shuffle baseline — search finds real signal"
        if self.z_score >= 1.0:
            return "marginal — observed exceeds shuffle mean by ~1σ"
        if self.z_score > -1.0:
            return (
                "indistinguishable from shuffle baseline — search finds "
                "no more signal on the real cipher than on a shuffle"
            )
        return "below shuffle mean — search underperforms its own null"

    def summary(self) -> str:
        return (
            f"SearchCalibrationResult (n_shuffles={self.n_shuffles}, "
            f"n_cipher_symbols={self.n_cipher_symbols})\n"
            f"  observed net_signal : {self.observed_net_signal:>7.1%}\n"
            f"  shuffle mean        : {self.shuffle_mean:>7.1%}\n"
            f"  shuffle std         : {self.shuffle_std:>7.1%}\n"
            f"  z-score             : {self.z_score:>7.2f}\n"
            f"  percentile          : {self.percentile:>5.1f}%  "
            "(100 = above all shuffles)\n"
            f"  Interpretation: {self._interpretation()}"
        )

    def to_dict(self) -> dict:
        return {
            "observed_net_signal": self.observed_net_signal,
            "shuffle_net_signals": list(self.shuffle_net_signals),
            "shuffle_mean": self.shuffle_mean,
            "shuffle_std": self.shuffle_std,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "n_shuffles": self.n_shuffles,
            "n_cipher_symbols": self.n_cipher_symbols,
        }
