"""Optional visualization helpers. Requires matplotlib (pip install dictcollision[viz]).

Replicates a subset of the paper's figures so users can drop their own
data in and get publication-style plots:

    plot_decomposition        paper Figure 1 (single-bar four-category)
    plot_size_sweep           paper Figure 2 (apparent vs net across sizes)
    plot_method_comparison    paper Figure 5 (six correction methods)
    plot_length_stratified    paper Figure 13 (length-stratified correction)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dictcollision._types import ClassifyResult, LengthBucket


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install dictcollision[viz]"
        ) from e
    return plt


def plot_decomposition(result: "ClassifyResult", title: str = "") -> None:
    """Plot the four-category stacked bar (replicates paper Figure 1)."""
    plt = _require_matplotlib()

    categories = ["Signal", "Shared hit", "Anti-signal", "Shared miss"]
    values = [
        result.signal,
        result.shared_hit,
        result.anti_signal,
        result.shared_miss,
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, values, color=colors)

    for bar, val in zip(bars, values):
        if val > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_ylabel("Fraction of tokens")
    ax.set_title(title or "Four-category token decomposition")
    ax.set_ylim(0, max(max(values), 0.01) * 1.2)
    ax.axhline(y=0, color="black", linewidth=0.5)

    net_text = f"Net signal: {result.net_signal:.1%}"
    app_text = f"Apparent hit rate: {result.apparent_hit_rate:.1%}"
    ax.text(
        0.98, 0.95,
        f"{app_text}\n{net_text}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


def plot_size_sweep(
    sizes: list[int],
    results: list["ClassifyResult"],
    title: str = "",
) -> None:
    """Paper Figure 2: apparent hit rate, net signal, and shared-hit noise
    across a dictionary-size sweep.

    Parameters
    ----------
    sizes : list of int
        X-axis values (dictionary sizes). Must match `len(results)`.
    results : list of ClassifyResult
        One result per size.
    title : str
    """
    plt = _require_matplotlib()
    if len(sizes) != len(results):
        raise ValueError("sizes and results must have the same length")

    apparent = [r.apparent_hit_rate * 100 for r in results]
    net = [r.net_signal * 100 for r in results]
    shared = [r.shared_hit * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, apparent, "o-", label="Apparent hit rate", color="#e67e22")
    ax.plot(sizes, net, "s-", label="Net signal", color="#2980b9")
    ax.plot(sizes, shared, "^--", label="Shared hits (noise)", color="#7f8c8d", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Dictionary size")
    ax.set_ylabel("Rate (%)")
    ax.set_title(title or "Effect of dictionary size on signal isolation")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_method_comparison(
    sizes: list[int],
    methods: dict[str, list[float]],
    title: str = "",
) -> None:
    """Paper Figure 5: six correction methods compared across dict sizes.

    Parameters
    ----------
    sizes : list of int
        X-axis values (dictionary sizes).
    methods : dict mapping method name -> list of fractions (one per size)
        Typical keys: "apparent_hit_rate", "subtract_null",
        "permutation_test", "bh_fdr", "blast_evalue", "four_category_net".
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    style = {
        "apparent_hit_rate": ("o-", "#e67e22"),
        "subtract_null": ("s--", "#c0392b"),
        "permutation_test": ("^--", "#f1c40f"),
        "bh_fdr": ("v--", "#27ae60"),
        "blast_evalue": ("D--", "#8e44ad"),
        "four_category_net": ("o-", "#2980b9"),
    }
    for name, values in methods.items():
        if len(values) != len(sizes):
            raise ValueError(f"method {name!r} has {len(values)} values, expected {len(sizes)}")
        fmt, color = style.get(name, ("o-", None))
        ax.plot(sizes, [v * 100 for v in values], fmt, color=color, label=name)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Dictionary size")
    ax.set_ylabel("Signal fraction of real tokens (%)")
    ax.set_title(title or "Correction methods compared")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_length_stratified(
    buckets: list["LengthBucket"],
    title: str = "",
) -> None:
    """Paper Figure 13: apparent hit rate, net signal, and correction gap
    as a function of decoded-token length."""
    plt = _require_matplotlib()

    lengths = [b.length for b in buckets]
    apparent = [b.apparent_hit_rate * 100 for b in buckets]
    net = [b.net_signal * 100 for b in buckets]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lengths, apparent, "o-", label="Apparent hit rate", color="#e67e22")
    ax.plot(lengths, net, "s-", label="Net signal", color="#2980b9")
    ax.fill_between(lengths, net, apparent, alpha=0.2, color="#3498db",
                    label="Correction magnitude")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Decoded token length (characters)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(title or "Framework necessity by token length")
    ax.legend()
    plt.tight_layout()
    plt.show()
