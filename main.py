"""Main file with GA main func and Argparse"""

import argparse
import warnings
import copy
from joblib import Parallel, delayed
import numpy as np
from data import X_train, X_test, y_train, y_test, FILENAME
from models import evaluate_baseline, fitness_function, DEFAULT_MODEL
from genetic_algorithm import (
    initialize_population,
    tournament_selection,
    crossover,
    mutation,
)

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments for the genetic algorithm configuration.

    Returns:

        argparse.Namespace:
            Parsed command-line arguments containing:
            - dataset (str): Path to the CSV dataset.
            - generations (int): Number of generations.
            - penalty (float): Penalty coefficient for using too many features.
            - population_size (int): Number of chromosomes in each generation.
            - mutation_rate (float): Probability of mutation during evolution.
            - model (str): Classifier type ('logistic' or 'random-forest').
            - scoring (str): Fitness evaluation metric.

    """
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for Feature Selection"
    )
    parser.add_argument(
        "--dataset",
        default=FILENAME,
        type=str,
        help=f"Path to the CSV dataset (default: {FILENAME})",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=150,
        help="Number of generations (default: 100)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.01,
        help="Penalty for using too many features (default: 0.05)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=30,
        help="Number of chromosomes per generation (default: 30)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.03,
        help="Probability of bit flip during mutation (default: 0.02)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "random-forest"],
        help="Classifier to use: logistic or random-forest (default: logistic)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "accuracy", "f1"],
        help="Scoring metric for fitness evaluation (default: roc_auc)",
    )
    return parser.parse_args()


#  два ключових параметри, підбирати супер акуратно і малнькими кроками
N_GENERATION = 30
PENALTY = 0.1
NO_IMPROVE_LIMIT = int(
    N_GENERATION * 1
)  # ліміт для зупинки алгоритмку у випадку, якщо не покращується метрика,
# ставити максимально великий, якщо ціль - найкраща метрика


def genetic_algorithm(
    X_train,
    y_train,
    feature_names,
    population_size: int = 15,
    # N_GENERATION: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    tournament_k: int = 3,
    # PENALTY: float = 0.01,
    cv: int = 3,
    scoring: str = "roc_auc",
    verbose: bool = True,
):
    """
    Run a genetic algorithm for feature selection.
    The algorithm evolves a population of chromosomes where each chromosome
    represents a subset of selected features. Fitness is evaluated using
    cross-validation and a chosen scoring metric.
    Args:
        X_train (pd.DataFrame or np.ndarray):
            Training feature matrix.
        y_train (pd.Series or np.ndarray):
            Target labels.
        feature_names (list[str]):
            List of feature names corresponding to columns in X_train.
        population_size (int, optional):
            Number of chromosomes in each generation.
            Larger values increase search diversity but require more computation.
            Recommended range: 15–100.
            Default is 15.
        crossover_rate (float, optional):
            Probability of crossover between parent chromosomes.
            Higher values encourage exploration of new feature combinations.
            Recommended range: 0.7–0.9.
            Default is 0.8.
        mutation_rate (float, optional):
            Probability of random mutation (bit flip).
            Small values help maintain stability while avoiding local minima.
            Recommended range: 0.01–0.1.
            Default is 0.05.
        tournament_k (int, optional):
            Number of individuals participating in tournament selection.
            Higher values increase selection pressure.
            Recommended range: 2–5.
            Default is 3.
        cv (int, optional):
            Number of cross-validation folds used during fitness evaluation.
            Higher values provide more reliable estimates but increase runtime.
            Recommended range: 3–10.
            Default is 3.
        scoring (str, optional):
            Metric used for fitness evaluation.
            Supported values:
            - "roc_auc"
            - "accuracy"
            - "f1"
            Default is "roc_auc".
        verbose (bool, optional):
            If True, prints progress and intermediate results during execution.
            Default is True.
    Notes:

        All major hyperparameters can be tuned depending on dataset size,
        computational resources, and optimization goals.
        General recommendations:
        - Increase population_size for larger or more complex datasets.
        - Increase mutation_rate if the algorithm converges too early.
        - Reduce mutation_rate if solutions become unstable.
        - Use higher cv values for more reliable fitness estimation.
        - Use "roc_auc" for imbalanced classification problems.
    Returns:
        dict:
            Dictionary containing the best chromosome, selected features,
            fitness history, and evaluation metrics.
    """

    no_improve_count = 0

    X_np = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    n_features = X_np.shape[1]
    feature_names = list(feature_names)

    population = initialize_population(population_size, n_features)
    best_chromosome = population[0].copy()
    best_fitness = -np.inf
    history = []

    for gen in range(N_GENERATION):
        # не паралельно
        # fitness_scores = np.array(
        #     [
        #         fitness_function(ch, X_np, y_np, model, PENALTY, cv, scoring)
        #         for ch in population
        #     ]
        # )

        # паралельно (швидше)
        fitness_scores = np.array(
            Parallel(n_jobs=-1)(
                delayed(fitness_function)(
                    ch, X_np, y_np, copy.deepcopy(DEFAULT_MODEL), PENALTY, cv, scoring
                )
                for ch in population
            )
        )
        gen_best_idx = np.argmax(fitness_scores)

        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        history.append(
            {
                "gen": gen + 1,
                "best_fitness": float(best_fitness),
                "avg_fitness": float(np.mean(fitness_scores)),
                "n_selected": int(best_chromosome.sum()),
            }
        )

        if no_improve_count >= NO_IMPROVE_LIMIT:
            print(f"Рання зупинка на генерації {gen+1}")
            break
        n_selected = int(best_chromosome.sum())
        if (gen + 1) % 2 == 0:
            color = "\033[31m"
        else:
            color = "\033[32m"
        if verbose:
            print(
                f"{color}Gen {gen+1:>3}/{N_GENERATION} | "
                f"best_fitness={best_fitness:.4f} | "
                f"avg_fitness={fitness_scores.mean():.4f} | "
                f"features={n_selected}/{n_features}\033[0m"
            )

        new_population = [best_chromosome.copy()]
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores, tournament_k)
            p2 = tournament_selection(population, fitness_scores, tournament_k)

            child1, child2 = crossover(p1, p2, crossover_rate)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)
    selected_features = [
        feature_names[i] for i, bit in enumerate(best_chromosome) if bit == 1
    ]
    return best_chromosome, selected_features, history


if __name__ == "__main__":
    args = parse_args()

    N_GENERATION = args.generations
    PENALTY = args.penalty

    print(f"Geneerations amount: {N_GENERATION}, \nPenalty: {PENALTY}, \nStop limit \
            {NO_IMPROVE_LIMIT}, \nModel: {DEFAULT_MODEL}")

    print("Start")

    base_model = DEFAULT_MODEL
    acc_base, f1_base, auc_base = evaluate_baseline(
        base_model, X_train, y_train, X_test, y_test
    )
    print("Baseline:")
    print(f"  Accuracy: {acc_base:.3f} | F1: {f1_base:.3f} | ROC-AUC: {auc_base:.3f}")

    model = DEFAULT_MODEL
    best_chromosome, selected_features, history = genetic_algorithm(
        X_train,
        y_train,
        X_train.columns,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        scoring=args.scoring,
    )

    eval_model = DEFAULT_MODEL
    acc, f1, auc = evaluate_baseline(
        eval_model,
        X_train[selected_features],
        y_train,
        X_test[selected_features],
        y_test,
    )
    print(f"GA Model -> Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Features: {len(X_train.columns)} → {len(selected_features)}")
    print(f"Selected: {selected_features}")
    print(f"START (all features):")
    print(f"  Accuracy: {acc_base:.3f} | F1: {f1_base:.3f} | ROC-AUC: {auc_base:.3f}")
    print(f"\nGA Model ({len(selected_features)} features):")
    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print("=" * 50)

    from visualization import plot_results

    delta_roc_auc = auc - auc_base
    delta_acc = acc - acc_base
    delta_f1 = f1 - f1_base
    plot_results(
        history,
        selected_features,
        X_train.columns.tolist(),
        delta_roc_auc,
        delta_acc,
        delta_f1,
        "LogisticRegression",
        PENALTY,
        5,
    )
