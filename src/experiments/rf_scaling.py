from __future__ import annotations
from src.bagging.random_forest import RandomForestClassifier
from src.metrics.evaluation import accuracy_score
from src.experiments.utils import get_dataset, plot_scaling_curve


def run_experiment_3() -> None:
    """Run Experiment 3: Random Forest scaling (estimators and depth variations)."""
    datasets = ["breast_cancer", "adult", "mnist"]
    print("================================================================================")
    print("Experiment 3: Random Forest Scaling")
    print("================================================================================")

    # Sequence of estimator counts to test
    n_estimators_list = [1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # Sequence of tree depths to test
    max_depths = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    for name in datasets:
        print(f"\nEvaluating Random Forest scaling on dataset: {name.upper()}")
        X_train, X_test, y_train, y_test = get_dataset(name)

        # ----------------------------------------------------------------------
        # Part (a): Vary n_estimators, max_depth=None
        # ----------------------------------------------------------------------
        print("Part (a): Varying n_estimators (max_depth=None)...")
        test_accs_est = []
        oob_scores_est = []

        for n in n_estimators_list:
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=None,
                bootstrap=True,
                oob_score=True,
                n_jobs=4,
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Test accuracy
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_accs_est.append(test_acc)

            # OOB score
            oob = model.oob_score_ if hasattr(model, "oob_score_") and model.oob_score_ is not None else 0.0
            oob_scores_est.append(oob)
            print(f"  n_estimators={n:<3}: test_acc={test_acc:.4f}, oob_score={oob:.4f}")

        # Plot Part (a)
        plot_scaling_curve(
            x_vals=n_estimators_list,
            train_vals=oob_scores_est,
            test_vals=test_accs_est,
            xlabel="Number of Estimators",
            ylabel="Accuracy",
            title=f"Random Forest Estimator Scaling on {name.upper()}",
            save_filename=f"rf_estimators_scaling_{name}.png",
            legend_labels=("OOB Accuracy", "Test Accuracy"),
        )
        print(f"Saved plot to figures/rf_estimators_scaling_{name}.png")

        # ----------------------------------------------------------------------
        # Part (b): Fix n_estimators=100, vary max_depth
        # ----------------------------------------------------------------------
        print("Part (b): Varying max_depth (n_estimators=100)...")
        test_accs_depth = []
        oob_scores_depth = []

        for d in max_depths:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=d,
                bootstrap=True,
                oob_score=True,
                n_jobs=4,
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Test accuracy
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_accs_depth.append(test_acc)

            # OOB score
            oob = model.oob_score_ if hasattr(model, "oob_score_") and model.oob_score_ is not None else 0.0
            oob_scores_depth.append(oob)
            print(f"  max_depth={d:<2}: test_acc={test_acc:.4f}, oob_score={oob:.4f}")

        # Plot Part (b)
        plot_scaling_curve(
            x_vals=max_depths,
            train_vals=oob_scores_depth,
            test_vals=test_accs_depth,
            xlabel="Max Depth",
            ylabel="Accuracy",
            title=f"Random Forest Depth Scaling on {name.upper()}",
            save_filename=f"rf_depth_scaling_{name}.png",
            legend_labels=("OOB Accuracy", "Test Accuracy"),
        )
        print(f"Saved plot to figures/rf_depth_scaling_{name}.png")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment_3()
