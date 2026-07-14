from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from src.trees.decision_tree import DecisionTree, DecisionStump
from src.metrics.evaluation import accuracy_score, precision_recall_f1
from src.experiments.utils import get_dataset


def run_experiment_1() -> None:
    """Run Experiment 1: Baseline comparison of DecisionTree, DecisionStump, and sklearn DecisionTree."""
    datasets = ["breast_cancer", "adult", "mnist"]
    print("================================================================================")
    print("Experiment 1: Baseline Comparison")
    print("================================================================================")

    for name in datasets:
        print(f"\nEvaluating dataset: {name.upper()}")
        X_train, X_test, y_train, y_test = get_dataset(name)

        # 1. Fit custom unpruned DecisionTree
        custom_tree = DecisionTree(criterion="gini", random_state=42)
        custom_tree.fit(X_train, y_train)
        custom_preds = custom_tree.predict(X_test)
        # Use predict_proba for AUC-ROC calculation (column 1 for class 1)
        custom_probs = custom_tree.predict_proba(X_test)
        
        # 2. Fit custom DecisionStump (max_depth=1)
        custom_stump = DecisionStump(criterion="gini", random_state=42)
        custom_stump.fit(X_train, y_train)
        custom_stump_preds = custom_stump.predict(X_test)
        custom_stump_probs = custom_stump.predict_proba(X_test)

        # 3. Fit sklearn DecisionTreeClassifier
        sklearn_tree = DecisionTreeClassifier(criterion="gini", random_state=42)
        sklearn_tree.fit(X_train, y_train)
        sklearn_preds = sklearn_tree.predict(X_test)
        sklearn_probs = sklearn_tree.predict_proba(X_test)

        # Evaluate Custom Tree
        c_acc = accuracy_score(y_test, custom_preds)
        _, _, c_f1 = precision_recall_f1(y_test, custom_preds, average="macro")
        # For binary classification, roc_auc_score expects probabilities of class 1
        c_auc = float(roc_auc_score(y_test, custom_probs[:, 1]))

        # Evaluate Custom Stump
        s_acc = accuracy_score(y_test, custom_stump_preds)
        _, _, s_f1 = precision_recall_f1(y_test, custom_stump_preds, average="macro")
        s_auc = float(roc_auc_score(y_test, custom_stump_probs[:, 1]))

        # Evaluate Sklearn Tree
        sk_acc = accuracy_score(y_test, sklearn_preds)
        _, _, sk_f1 = precision_recall_f1(y_test, sklearn_preds, average="macro")
        sk_auc = float(roc_auc_score(y_test, sklearn_probs[:, 1]))

        # Printing results table
        print(f"{'Classifier':<25} | {'Accuracy':<10} | {'Macro F1':<10} | {'AUC-ROC':<10}")
        print("-" * 65)
        print(f"{'Custom DecisionTree':<25} | {f'{c_acc:.4f}':<10} | {f'{c_f1:.4f}':<10} | {f'{c_auc:.4f}':<10}")
        print(f"{'Custom DecisionStump':<25} | {f'{s_acc:.4f}':<10} | {f'{s_f1:.4f}':<10} | {f'{s_auc:.4f}':<10}")
        print(f"{'Sklearn DecisionTree':<25} | {f'{sk_acc:.4f}':<10} | {f'{sk_f1:.4f}':<10} | {f'{sk_auc:.4f}':<10}")
        
        # Verify agreement within 2% absolute accuracy
        diff = abs(c_acc - sk_acc)
        print(f"Absolute Accuracy Difference (Custom vs Sklearn): {diff:.4f}")
        if diff <= 0.02:
            print("=> VERIFICATION PASSED: Agreement is within the 2% threshold.")
        else:
            print("=> VERIFICATION WARNING: Agreement exceeds the 2% threshold.")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment_1()
