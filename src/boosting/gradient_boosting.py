import numpy as np


class _RegressionNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class _RegressionTree:
    """Minimal regression tree (variance reduction), used as GBM's weak learner."""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, residuals):
        self.root = self._grow(X, residuals, depth=0)
        return self

    def _grow(self, X, r, depth):
        n = len(r)
        if depth >= self.max_depth or n < self.min_samples_split or np.allclose(r, r[0]):
            return _RegressionNode(value=r.mean())

        feature, threshold, gain = self._best_split(X, r)
        if feature is None or gain <= 0:
            return _RegressionNode(value=r.mean())

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._grow(X[left_mask], r[left_mask], depth + 1)
        right = self._grow(X[right_mask], r[right_mask], depth + 1)
        return _RegressionNode(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, r):
        n_samples, n_features = X.shape
        parent_var = r.var() * n_samples
        best_gain, best_feature, best_threshold = -1, None, None

        for feature in range(n_features):
            order = np.argsort(X[:, feature])
            xs, rs = X[order, feature], r[order]
            for i in range(1, n_samples):
                if xs[i] == xs[i - 1]:
                    continue
                left_r, right_r = rs[:i], rs[i:]
                left_var = left_r.var() * len(left_r)
                right_var = right_r.var() * len(right_r)
                gain = parent_var - left_var - right_var
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (xs[i] + xs[i - 1]) / 2
        return best_feature, best_threshold, best_gain

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class GradientBoostingClassifier:
    """Binary GBM with log-loss (binomial deviance), from scratch."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.init_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        p = y.mean()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        self.init_ = np.log(p / (1 - p))
        F = np.full(len(y), self.init_)

        self.trees_ = []
        for m in range(self.n_estimators):
            residuals = y - _sigmoid(F)
            tree = _RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        p1 = _sigmoid(F)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)