"""
Visualization for Genetic Algorithm results.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import FILENAME
from models import DEFAULT_MODEL
import os
from main import PENALTY

os.makedirs("results", exist_ok=True)

name = os.path.splitext(os.path.basename(FILENAME))[0]


def plot_results(
    history,
    selected_features,
    all_features,
    delta_roc_auc,
    delta_acc,
    delta_f1,
    model_name,
    penalty,
    cv,
):
    if not history:
        print("No history to plot.")
        return

    gens = [h["gen"] for h in history]
    best_fit = [h["best_fitness"] for h in history]
    avg_fit = [h["avg_fitness"] for h in history]
    n_selected = [h["n_selected"] for h in history]

    n_total = len(all_features)
    n_final_selected = len(selected_features)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Genetic Algorithm — Feature Selection Results",
        fontsize=15,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # =========================
    # Fitness plot
    # =========================
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(
        gens,
        best_fit,
        color="#2980b9",
        linewidth=2,
        label="Best fitness",
    )

    ax1.plot(
        gens,
        avg_fit,
        color="#e67e22",
        linewidth=1.5,
        linestyle="--",
        label="Avg fitness",
    )

    ax1.fill_between(
        gens,
        avg_fit,
        best_fit,
        alpha=0.12,
        color="#2980b9",
    )

    ax1.set_title("Fitness over Generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (score − penalty)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # =========================
    # Selected features plot
    # =========================
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.plot(
        gens,
        n_selected,
        color="#8e44ad",
        linewidth=2,
    )

    ax2.axhline(
        n_total,
        color="#aaa",
        linestyle=":",
        linewidth=1,
        label=f"Initial features ({n_total})",
    )

    ax2.set_title("Selected Features over Generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("# Features")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # =========================
    # Experiment summary block
    # =========================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    model_name = str(DEFAULT_MODEL).split("(")[0]

    summary_text = (
        f"GENETIC ALGORITHM SUMMARY\n\n"
        f"Dataset              : {name}\n"
        f"Model                : LogisticRegression\n"
        f"Generations          : {len(history)}\n"
        f"Penalty              : {PENALTY}\n"
        f"K-Fold CV            : 5\n\n"
        f"Feature Reduction\n"
        f"------------------------------\n"
        f"Initial features     : {n_total}\n"
        f"Selected features    : {n_final_selected}\n\n"
        f"Performance Change\n"
        f"------------------------------\n"
        f"Δ ROC-AUC            : {delta_roc_auc:+.4f}\n"
        f"Δ Accuracy           : {delta_acc:+.4f}\n"
        f"Δ F1-score           : {delta_f1:+.4f}"
    )

    ax3.text(
        0.03,
        0.97,
        summary_text,
        fontsize=11,
        va="top",
        family="monospace",
        color="#2c3e50",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#f8f9fa",
            edgecolor="#34495e",
            linewidth=2,
        ),
    )
    plt.savefig(
        f"results/ga_results_{name}.png",
        dpi=150,
        bbox_inches="tight",
    )

    print(f"\nVisualization saved to results/ga_results_{name}.png")

    plt.show()
