"""
Visualization for Genetic Algorithm results.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import FILENAME

import os

os.makedirs("results", exist_ok=True)

name = os.path.splitext(os.path.basename(FILENAME))[0]


def plot_results(
    history: list[dict],
    selected_features: list[str],
    all_features: list[str],
) -> None:
    """
    Plots GA results after the algorithm finishes:
      1. Fitness over generations (best & average)
      2. Number of selected features over generations
      3. Bar chart of selected vs. dropped features

    Args:
        history:          list of dicts returned by genetic_algorithm()
                          keys: gen, best_fitness, avg_fitness, n_selected
        selected_features: feature names chosen by the best chromosome
        all_features:     full list of feature names before selection
    """
    if not history:
        print("No history to plot.")
        return

    gens = [h["gen"] for h in history]
    best_fit = [h["best_fitness"] for h in history]
    avg_fit = [h["avg_fitness"] for h in history]
    n_selected = [h["n_selected"] for h in history]
    n_total = len(all_features)

    selected_set = set(selected_features)
    feature_labels = [f[:25] + "…" if len(f) > 25 else f for f in all_features]
    colors = ["#2ecc71" if f in selected_set else "#e74c3c" for f in all_features]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Genetic Algorithm — Feature Selection Results", fontsize=15, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])  # full top row
    ax1.plot(gens, best_fit, color="#2980b9", linewidth=2, label="Best fitness")
    ax1.plot(
        gens,
        avg_fit,
        color="#e67e22",
        linewidth=1.5,
        linestyle="--",
        label="Avg fitness",
    )
    ax1.fill_between(gens, avg_fit, best_fit, alpha=0.12, color="#2980b9")
    ax1.set_title("Fitness over Generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (score − penalty)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(gens, n_selected, color="#8e44ad", linewidth=2)
    ax2.axhline(
        n_total, color="#aaa", linestyle=":", linewidth=1, label=f"Total ({n_total})"
    )
    ax2.set_title("Selected Features over Generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("# Features")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    y_pos = range(len(all_features))
    ax3.barh(
        y_pos, [1] * len(all_features), color=colors, edgecolor="white", height=0.8
    )
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels(feature_labels, fontsize=max(5, 9 - len(all_features) // 15))
    ax3.set_xticks([])
    ax3.set_title(
        f"Features: {len(selected_features)} selected / {n_total - len(selected_features)} dropped"
    )
    ax3.invert_yaxis()
    plt.savefig(f"results/ga_results_{name}.png", dpi=150, bbox_inches="tight")
    print("\nВізуалізацію збережено у ga_results.png")
    plt.show()
