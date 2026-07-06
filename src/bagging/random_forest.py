# Random Forest – placeholder stub
import numpy as np
from src.trees.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _resolve_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features is None:
            return n_features
        else:
            return self.max_features

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_feat = self._resolve_max_features(n_features)
        rng = np.random.default_rng(self.random_state)

        self.trees = []

        for i in range(self.n_estimators):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            values, counts = np.unique(votes, return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        return np.array(final_preds)