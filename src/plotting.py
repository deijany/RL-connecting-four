"""Comparison plots for the notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt

def plot_winrates(results: dict, colors: dict = None, total: int = None):
    """Grouped win/loss/tie bars per method.
    results: {method: {win, loss, tie}}
    colors:  single dict with keys "win", "loss", "tie" for bar colors.
    total:   if given, sets y-axis to [0, total] so bars show fraction of all games.
    """
    if colors is None:
        colors = {}
    methods = list(results)
    width = 0.25
    cat_colors = {"win": colors.get("win", "steelblue"),
                  "loss": colors.get("loss", "red"),
                  "tie": colors.get("tie", "gray")}
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, method in enumerate(methods):
        base = j * (3 * width + 0.05)
        ax.bar(base + 0 * width, results[method]["win"],  width=width, color=cat_colors["win"])
        ax.bar(base + 1 * width, results[method]["loss"], width=width, color=cat_colors["loss"])
        ax.bar(base + 2 * width, results[method]["tie"],  width=width, color=cat_colors["tie"])
    sorted_wins = sorted(results.values(), key=lambda d: d["win"], reverse=True)
    ref = sorted_wins[1]["win"] if len(sorted_wins) >= 2 else None
    if ref is not None:
        ax.axhline(ref, color="black", linestyle="--", linewidth=0.9)
        for j, method in enumerate(methods):
            wins = results[method]["win"]
            if wins == ref:
                continue
            pct = (wins - ref) / ref * 100
            label = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
            bar_x = j * (3 * width + 0.05)
            ax.text(bar_x, wins + 5, label, ha="center", va="bottom", fontsize=8)
    if total is not None:
        ax.set_ylim(0, total)
    centers = [j * (3 * width + 0.05) + width for j in range(len(methods))]
    ax.set_xticks(centers)
    ax.set_xticklabels(methods)
    ax.set_ylabel(f"Games (out of {total})" if total is not None else "Games")
    ax.set_title("Performance")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_colors.values()]
    ax.legend(handles, list(cat_colors.keys()), fontsize=8)
    fig.tight_layout()
    return fig


def plot_training_curves(curves: dict, colors: dict = None):
    """Loss curves per method on a shared environment-steps x-axis.
    curves: {method: (steps_list, losses_list)} or {method: losses_list}.
    colors: single dict with method names as keys for line colors.
    """
    if colors is None:
        colors = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, data in curves.items():
        if isinstance(data, tuple):
            xs, ys = data
        else:
            xs, ys = list(range(1, len(data) + 1)), data
        ax.plot(xs, ys, label=method, color=colors.get(method, None))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves (same env-step budget)")
    ax.legend()
    fig.tight_layout()
    return fig
