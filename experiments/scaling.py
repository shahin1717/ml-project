# Experiment 1: Baseline single tree vs sklearn reference – placeholder stub
from src.utils.preprocessing import load_breast_cancer, train_test_split, standardize
from src.trees.decision_tree import DecisionTree
from src.boosting.adaboost import AdaBoost
from src.bagging.random_forest import RandomForest
from src.metrics.evaluation import accuracy, f1_score
from sklearn.tree import DecisionTreeClassifier


def run_baseline():
    X, y = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s = standardize(X_train, X_test)

    results = {}

    tree = DecisionTree(max_depth=20, min_samples_split=2)
    tree.fit(X_train_s, y_train)
    preds = tree.predict(X_test_s)
    results["DecisionTree"] = {
        "accuracy": accuracy(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    stump = DecisionTree(max_depth=1)
    stump.fit(X_train_s, y_train)
    stump_preds = stump.predict(X_test_s)
    results["DecisionStump"] = {
        "accuracy": accuracy(y_test, stump_preds),
        "f1": f1_score(y_test, stump_preds),
    }

    sk_tree = DecisionTreeClassifier(random_state=42)
    sk_tree.fit(X_train_s, y_train)
    sk_preds = sk_tree.predict(X_test_s)
    results["sklearn_DecisionTree"] = {
        "accuracy": accuracy(y_test, sk_preds),
        "f1": f1_score(y_test, sk_preds),
    }

    ada = AdaBoost(n_estimators=50)
    ada.fit(X_train_s, y_train)
    ada_preds = ada.predict(X_test_s)
    results["AdaBoost"] = {
        "accuracy": accuracy(y_test, ada_preds),
        "f1": f1_score(y_test, ada_preds),
    }

    rf = RandomForest(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_preds = rf.predict(X_test_s)
    results["RandomForest"] = {
        "accuracy": accuracy(y_test, rf_preds),
        "f1": f1_score(y_test, rf_preds),
    }

    return results


if __name__ == "__main__":
    results = run_baseline()
    for model_name, metrics in results.items():
        print(f"{model_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")