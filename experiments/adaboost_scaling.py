# Experiment 2: AdaBoost scaling (1-200 stumps) – placeholder stub
from src.utils.preprocessing import load_breast_cancer, train_test_split, standardize
from src.boosting.adaboost import AdaBoost
from src.metrics.evaluation import accuracy
import matplotlib.pyplot as plt

def run_adaboost_scaling():
    X, y = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s = standardize(X_train, X_test)

    n_estimators_list = list(range(1, 201, 10))
    train_accuracies = []
    test_accuracies = []

    for n in n_estimators_list:
        model = AdaBoost(n_estimators=n)
        model.fit(X_train_s, y_train)

        train_preds = model.predict(X_train_s)
        test_preds = model.predict(X_test_s)

        train_acc = accuracy(y_train, train_preds)
        test_acc = accuracy(y_test, test_preds)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"n_estimators={n}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    return n_estimators_list, train_accuracies, test_accuracies

def plot_results(n_estimators_list, train_accuracies, test_accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(n_estimators_list, train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(n_estimators_list, test_accuracies, label="Test Accuracy", marker="o")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title("AdaBoost: Accuracy vs Number of Estimators")
    plt.legend()
    plt.grid(True)
    plt.savefig("report/figures/adaboost_scaling.png")
    plt.show()

if __name__ == "__main__":
    n_est, train_acc, test_acc = run_adaboost_scaling()
    plot_results(n_est, train_acc, test_acc)