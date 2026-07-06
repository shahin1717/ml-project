# AdaBoost (SAMME) – placeholder stub
import numpy as np
from src.trees.decision_tree import DecisionTree


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(y)
        y_signed = np.where(y == 0, -1, 1)

        sample_weight = np.ones(n_samples) / n_samples

        self.stumps = []
        self.alphas = []

        for t in range(self.n_estimators):
            stump = DecisionTree(max_depth=1)
            stump.fit(X, y, sample_weight=sample_weight)

            pred = stump.predict(X)
            pred_signed = np.where(pred == 0, -1, 1)

            incorrect = (pred_signed != y_signed)
            error = np.sum(sample_weight[incorrect]) / np.sum(sample_weight)

            error = np.clip(error, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)

            sample_weight = sample_weight * np.exp(-alpha * y_signed * pred_signed)
            sample_weight = sample_weight / np.sum(sample_weight)

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = []
        for stump, alpha in zip(self.stumps, self.alphas):
            pred = stump.predict(X)
            pred_signed = np.where(pred == 0, -1, 1)
            stump_preds.append(alpha * pred_signed)

        weighted_sum = np.sum(stump_preds, axis=0)
        final_signed = np.sign(weighted_sum)
        return np.where(final_signed == -1, 0, 1)