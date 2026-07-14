# # Experiment 6: Bias-variance decomposition (100 bootstrap replicates) – placeholder stub
# import numpy as np
# from src.utils.preprocessing import load_breast_cancer, train_test_split, standardize
# from src.trees.decision_tree import DecisionTree
# from src.boosting.adaboost import AdaBoostClassifier
# from src.bagging.random_forest import RandomForestClassifier


# def bias_variance_decomposition(model_fn, X_train, y_train, X_test, y_test, n_bootstraps=100, random_state=42):
#     rng = np.random.default_rng(random_state)
#     n_train = X_train.shape[0]
#     n_test = X_test.shape[0]

#     all_preds = np.zeros((n_bootstraps, n_test))

#     for b in range(n_bootstraps):
#         idx = rng.integers(0, n_train, size=n_train)
#         X_boot, y_boot = X_train[idx], y_train[idx]

#         model = model_fn()
#         model.fit(X_boot, y_boot)
#         all_preds[b] = model.predict(X_test)

#     main_prediction = np.round(all_preds.mean(axis=0))

#     bias_sq = np.mean((main_prediction - y_test) ** 2)
#     variance = np.mean(np.var(all_preds, axis=0))

#     return bias_sq, variance


# def run_experiment_6():
#     X, y = load_breast_cancer()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train_s, X_test_s = standardize(X_train, X_test)

#     models = {
#         "DecisionTree": lambda: DecisionTree(max_depth=10, random_state=42),
#         "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
#         "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
#     }

#     results = {}
#     for name, model_fn in models.items():
#         bias_sq, variance = bias_variance_decomposition(
#             model_fn, X_train_s, y_train, X_test_s, y_test, n_bootstraps=100
#         )
#         results[name] = {"bias_sq": bias_sq, "variance": variance}
#         print(f"{name}: bias^2={bias_sq:.4f}, variance={variance:.4f}")

#     return results


# if __name__ == "__main__":
#     run_experiment_6()

import numpy as np
import matplotlib.pyplot as plt

from src.utils.preprocessing import (
    load_breast_cancer,
    train_test_split,
    standardize,
)
from src.trees.decision_tree import DecisionTree
from src.boosting.adaboost import AdaBoostClassifier
from src.bagging.random_forest import RandomForestClassifier


def bias_variance_decomposition(
    model_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    n_bootstraps=100,
    random_state=42,
):
    """
    Estimate Bias² and Variance using bootstrap replicates.

    Following the idea of Breiman (1996):
        - Bootstrap training sets
        - Evaluate on one fixed test set
        - Use average predicted probability
    """

    rng = np.random.default_rng(random_state)

    n_train = len(X_train)
    n_test = len(X_test)

    # probability of positive class
    predictions = np.zeros((n_bootstraps, n_test))

    for b in range(n_bootstraps):

        indices = rng.integers(0, n_train, size=n_train)

        X_boot = X_train[indices]
        y_boot = y_train[indices]

        model = model_fn()
        model.fit(X_boot, y_boot)

        probs = model.predict_proba(X_test)

        # probability of class 1
        predictions[b] = probs[:, 1]

    mean_prediction = predictions.mean(axis=0)

    bias_sq = np.mean((mean_prediction - y_test) ** 2)

    variance = np.mean(np.var(predictions, axis=0))

    return bias_sq, variance


def plot_results(results):

    models = list(results.keys())

    bias = [results[m]["bias_sq"] for m in models]
    variance = [results[m]["variance"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8,5))

    plt.bar(x - width/2, bias, width, label="Bias²")
    plt.bar(x + width/2, variance, width, label="Variance")

    plt.xticks(x, models)

    plt.ylabel("Value")
    plt.title("Bias-Variance Decomposition")

    plt.legend()

    plt.tight_layout()

    plt.show()


def run_experiment_6():

    X, y = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test = standardize(X_train, X_test)

    models = {
        "DecisionTree": lambda: DecisionTree(
            max_depth=10,
            random_state=42,
        ),

        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=100,
            random_state=42,
        ),

        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        ),
    }

    results = {}

    print("=" * 55)
    print("Bias-Variance Decomposition (B = 100)")
    print("=" * 55)

    for name, model_fn in models.items():

        bias_sq, variance = bias_variance_decomposition(
            model_fn,
            X_train,
            y_train,
            X_test,
            y_test,
            n_bootstraps=100,
        )

        results[name] = {
            "bias_sq": bias_sq,
            "variance": variance,
        }

        print(
            f"{name:<15}"
            f" Bias² = {bias_sq:.4f}"
            f"   Variance = {variance:.4f}"
        )

    print("=" * 55)

    plot_results(results)

    return results


if __name__ == "__main__":
    run_experiment_6()