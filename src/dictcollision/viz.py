"""Optional visualization helpers. Requires matplotlib (pip install dictcollision[viz])."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dictcollision._types import ClassifyResult


def plot_decomposition(result: "ClassifyResult", title: str = "") -> None:
    """Plot the four-category stacked bar (replicates paper Figure 1).

    Parameters
    ----------
    result : ClassifyResult
        Output of dictcollision.classify().
    title : str
        Optional plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install dictcollision[viz]"
        ) from e

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
        0.98,
        0.95,
        f"{app_text}\n{net_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()
