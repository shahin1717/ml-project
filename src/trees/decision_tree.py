# Decision Tree (CART) – placeholder stub
import numpy as np 


def gini(y, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    classes = np.unique(y)
    total_weight = np.sum(sample_weight)
    impurity = 1.0
    for c in classes:
        mask = (y == c)
        p = np.sum(sample_weight[mask]) / total_weight
        impurity -= p ** 2
    return impurity


def information_gain(y, y_left, y_right, w=None, w_left=None, w_right=None):
    parent_impurity = gini(y, w)
    total_w = np.sum(w) if w is not None else len(y)
    w_left_sum = np.sum(w_left) if w_left is not None else len(y_left)
    w_right_sum = np.sum(w_right) if w_right is not None else len(y_right)
    weight_left = w_left_sum / total_w
    weight_right = w_right_sum / total_w
    child_impurity = weight_left * gini(y_left, w_left) + weight_right * gini(y_right, w_right)
    return parent_impurity - child_impurity


def best_split(X, y, sample_weight=None, max_features=None, rng=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    if rng is None:
        rng = np.random.default_rng()

    n_features = X.shape[1]

    if max_features is None:
        feature_indices = np.arange(n_features)
    else:
        feature_indices = rng.choice(n_features, size=max_features, replace=False)

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in feature_indices:
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_mask = X[:, feature] <= t
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            gain = information_gain(
                y, y[left_mask], y[right_mask],
                sample_weight, sample_weight[left_mask], sample_weight[right_mask]
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gain
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.root = self._grow_tree(X, y, sample_weight, depth=0)

    def _grow_tree(self, X, y, sample_weight, depth):
        n_samples = len(y)
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            leaf_value = self._most_common_label(y, sample_weight)
            return Node(value=leaf_value)

        feature, threshold, gain = best_split(X, y, sample_weight, self.max_features, self.rng)

        if feature is None or gain <= 0:
            leaf_value = self._most_common_label(y, sample_weight)
            return Node(value=leaf_value)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _most_common_label(self, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        classes = np.unique(y)
        weighted_counts = [np.sum(sample_weight[y == c]) for c in classes]
        return classes[np.argmax(weighted_counts)]
    

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)