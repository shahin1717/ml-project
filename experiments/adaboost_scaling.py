from __future__ import annotations
from src.boosting.adaboost import AdaBoostClassifier
from src.metrics.evaluation import accuracy_score
from experiments.utils import get_dataset, plot_scaling_curve


def run_experiment_2() -> None:
    """Run Experiment 2: AdaBoost scaling (train and test errors vs. n_estimators)."""
    datasets = ["breast_cancer", "adult", "mnist"]
    print("================================================================================")
    print("Experiment 2: AdaBoost Scaling")
    print("================================================================================")

    for name in datasets:
        print(f"\nTraining AdaBoost on dataset: {name.upper()}")
        X_train, X_test, y_train, y_test = get_dataset(name)

        # Fit single model with 200 estimators
        model = AdaBoostClassifier(n_estimators=200, criterion="gini", random_state=42)
        model.fit(X_train, y_train)

        n_fitted = len(model.estimators_)
        print(f"Fitted {n_fitted} estimators (early stopping may have occurred).")

        # Collect train and test errors
        train_errors = []
        test_errors = []

        for train_pred in model.staged_predict(X_train):
            acc = accuracy_score(y_train, train_pred)
            train_errors.append(1.0 - acc)

        for test_pred in model.staged_predict(X_test):
            acc = accuracy_score(y_test, test_pred)
            test_errors.append(1.0 - acc)

        x_vals = list(range(1, len(train_errors) + 1))
        
        # Plot and save
        plot_scaling_curve(
            x_vals=x_vals,
            train_vals=train_errors,
            test_vals=test_errors,
            xlabel="Number of Estimators",
            ylabel="Error Rate",
            title=f"AdaBoost Scaling Error on {name.upper()}",
            save_filename=f"adaboost_scaling_{name}.png",
            legend_labels=("Train Error", "Test Error"),
        )
        print(f"Saved plot to figures/adaboost_scaling_{name}.png")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment_2()
