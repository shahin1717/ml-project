"""Decision Tree (CART) – full implementation.

Implements a binary Decision Tree classifier following the CART methodology,
as required by §2.1 of ML_FINAL_PROJECT.tex.

Public API
----------
DecisionTree(max_depth, min_samples_split, criterion, max_features, random_state)
    .fit(X, y, sample_weight) -> DecisionTree
    .predict(X)               -> np.ndarray
    .predict_proba(X)         -> np.ndarray   shape (n_samples, n_classes)
    .depth                    -> int           (property)
    .n_leaves                 -> int           (property)
    .feature_importances()    -> np.ndarray   shape (n_features,)
    .__repr__()               -> str
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Impurity functions
# ---------------------------------------------------------------------------

def gini(y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    """Weighted Gini impurity: I_G = 1 - sum_c p_c^2."""
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    total_weight = sample_weight.sum()
    if total_weight == 0:
        return 0.0
    classes = np.unique(y)
    impurity = 1.0
    for c in classes:
        p = sample_weight[y == c].sum() / total_weight
        impurity -= p * p
    return float(impurity)


def entropy(
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Weighted Shannon entropy: I_E = -sum_c p_c * log2(p_c + eps)."""
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    total_weight = sample_weight.sum()
    if total_weight == 0:
        return 0.0
    classes = np.unique(y)
    h = 0.0
    for c in classes:
        p = sample_weight[y == c].sum() / total_weight
        h -= p * math.log2(p + eps)
    return float(h)


def _impurity_fn(criterion: str):
    """Return the impurity callable for the given criterion string."""
    if criterion == "gini":
        return gini
    if criterion == "entropy":
        return entropy
    raise ValueError(f"Unknown criterion '{criterion}'. Use 'gini' or 'entropy'.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _class_distribution(
    y: np.ndarray,
    sample_weight: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Normalized weighted class counts as a probability vector.

    Parameters
    ----------
    y             : shape (n,)
    sample_weight : shape (n,)
    classes       : global class list, shape (n_classes,)

    Returns
    -------
    np.ndarray of shape (n_classes,), sums to 1.
    """
    counts = np.array([sample_weight[y == c].sum() for c in classes], dtype=float)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def _resolve_max_features(max_features, n_features: int) -> int:
    """Convert max_features to a concrete integer count."""
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(math.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(math.log2(n_features)))
    if isinstance(max_features, int):
        return max(1, min(max_features, n_features))
    raise ValueError(
        f"max_features must be None, 'sqrt', 'log2', or int — got {max_features!r}"
    )


# ---------------------------------------------------------------------------
# O(N log N) incremental impurity helpers
# ---------------------------------------------------------------------------

def _gini_from_counts(counts: dict, total_w: float) -> float:
    """Gini impurity from a {class: weight} dict."""
    if total_w <= 0:
        return 0.0
    impurity = 1.0
    for w in counts.values():
        p = w / total_w
        impurity -= p * p
    return impurity


def _entropy_from_counts(counts: dict, total_w: float, eps: float = 1e-12) -> float:
    """Shannon entropy from a {class: weight} dict."""
    if total_w <= 0:
        return 0.0
    h = 0.0
    for w in counts.values():
        p = w / total_w
        h -= p * math.log2(p + eps)
    return h


def _impurity_from_counts(counts: dict, total_w: float, criterion: str) -> float:
    if criterion == "gini":
        return _gini_from_counts(counts, total_w)
    return _entropy_from_counts(counts, total_w)


# ---------------------------------------------------------------------------
# Best split (O(N log N) per feature)
# ---------------------------------------------------------------------------

def best_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    criterion: str = "gini",
    max_features: int | str | None = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[int], Optional[float], float]:
    """Find the best (feature, threshold, gain) using an incremental sweep.

    Thresholds are midpoints between consecutive distinct feature values (DT.3).
    The sweep is O(N log N) per feature — no repeated O(N) rescans.

    Returns
    -------
    (best_feature, best_threshold, best_gain)
    best_feature is None when no improving split exists.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_features = X.shape
    k = _resolve_max_features(max_features, n_features)

    if k >= n_features:
        feature_indices = np.arange(n_features)
    else:
        feature_indices = rng.choice(n_features, size=k, replace=False)

    classes = np.unique(y)
    total_w = sample_weight.sum()
    parent_impurity = _impurity_fn(criterion)(y, sample_weight)

    best_gain: float = -1.0
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None

    for feature in feature_indices:
        order = np.argsort(X[:, feature], kind="mergesort")
        xs = X[order, feature]
        ys = y[order]
        ws = sample_weight[order]

        # Everything starts on the right
        left_counts: dict = {c: 0.0 for c in classes}
        right_counts: dict = {c: float(ws[ys == c].sum()) for c in classes}
        left_w = 0.0
        right_w = total_w

        for i in range(n_samples - 1):
            c = ys[i]
            left_counts[c] += ws[i]
            right_counts[c] -= ws[i]
            left_w += ws[i]
            right_w -= ws[i]

            # Evaluate a threshold only between *distinct* consecutive values
            if xs[i] == xs[i + 1]:
                continue

            t = (xs[i] + xs[i + 1]) / 2.0

            left_imp = _impurity_from_counts(left_counts, left_w, criterion)
            right_imp = _impurity_from_counts(right_counts, right_w, criterion)
            gain = (
                parent_impurity
                - (left_w / total_w) * left_imp
                - (right_w / total_w) * right_imp
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = int(feature)
                best_threshold = float(t)

    return best_feature, best_threshold, best_gain


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """A single node in the decision tree.

    Every node — internal or leaf — carries:
      value             : normalized class-probability vector (n_classes,)
      samples           : number of training samples that reached this node
      impurity          : impurity value at this node
      impurity_decrease : ΔI gained by this split (0 for leaves)
      is_leaf_          : explicit leaf flag (do NOT infer from value being None)
    """

    __slots__ = (
        "feature",
        "threshold",
        "left",
        "right",
        "value",
        "samples",
        "impurity",
        "impurity_decrease",
        "is_leaf_",
    )

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[np.ndarray] = None,
        samples: int = 0,
        impurity: float = 0.0,
        impurity_decrease: float = 0.0,
        is_leaf: bool = False,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value           # shape (n_classes,), normalized class distribution
        self.samples = samples
        self.impurity = impurity
        self.impurity_decrease = impurity_decrease
        self.is_leaf_ = is_leaf


# ---------------------------------------------------------------------------
# DecisionTree
# ---------------------------------------------------------------------------

class DecisionTree:
    """Binary Decision Tree classifier (CART).

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree.  None = grow until pure/other stopping condition.
    min_samples_split : int
        Minimum samples required to attempt a split.
    criterion : {"gini", "entropy"}
        Splitting criterion.
    max_features : int, "sqrt", "log2", or None
        Number of features to consider at each split.  None = all features.
    random_state : int or None
        Seed for the internal random generator (reproducibility).
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionTree":
        """Train the decision tree.

        Parameters
        ----------
        X             : shape (n_samples, n_features)
        y             : shape (n_samples,)
        sample_weight : shape (n_samples,) or None — defaults to uniform weights

        Returns
        -------
        self
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.rng = np.random.default_rng(self.random_state)   # reset for reproducibility
        self.root = self._grow_tree(X, y, sample_weight, depth=0)
        return self

    # ------------------------------------------------------------------
    # _grow_tree
    # ------------------------------------------------------------------

    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
    ) -> Node:
        n_samples = len(y)
        class_dist = _class_distribution(y, sample_weight, self.classes_)
        node_impurity = _impurity_fn(self.criterion)(y, sample_weight)

        def make_leaf() -> Node:
            return Node(
                value=class_dist,
                samples=n_samples,
                impurity=node_impurity,
                is_leaf=True,
            )

        # --- Stopping criteria (DT.4) ---
        if len(np.unique(y)) == 1:
            return make_leaf()

        if n_samples < self.min_samples_split:
            return make_leaf()

        if self.max_depth is not None and depth >= self.max_depth:
            return make_leaf()

        # All feature vectors identical → no valid split exists
        if np.unique(X, axis=0).shape[0] == 1:
            return make_leaf()

        # --- Best split ---
        feature, threshold, gain = best_split(
            X, y, sample_weight,
            criterion=self.criterion,
            max_features=self.max_features,
            rng=self.rng,
        )

        if feature is None or gain <= 0:
            return make_leaf()

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._grow_tree(
            X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1
        )
        right = self._grow_tree(
            X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1
        )

        return Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            value=class_dist,
            samples=n_samples,
            impurity=node_impurity,
            impurity_decrease=gain,
            is_leaf=False,
        )

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        return np.array([self._traverse_proba(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _traverse_proba(self, x: np.ndarray, node: Node) -> np.ndarray:
        if node.is_leaf_:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)

    # ------------------------------------------------------------------
    # Introspection properties
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Maximum depth of the fitted tree."""
        return self._node_depth(self.root)

    def _node_depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf_:
            return 0
        return 1 + max(self._node_depth(node.left), self._node_depth(node.right))

    @property
    def n_leaves(self) -> int:
        """Total number of leaf nodes."""
        return self._count_leaves(self.root)

    def _count_leaves(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        if node.is_leaf_:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    # ------------------------------------------------------------------
    # Feature importances (MDI)
    # ------------------------------------------------------------------

    def feature_importances(self) -> np.ndarray:
        """Normalized total impurity reduction per feature (MDI).

        Returns
        -------
        np.ndarray of shape (n_features,), summing to 1.
        """
        importances = np.zeros(self.n_features_, dtype=float)
        self._accumulate_importance(self.root, importances)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _accumulate_importance(
        self, node: Optional[Node], importances: np.ndarray
    ) -> None:
        if node is None or node.is_leaf_:
            return
        importances[node.feature] += node.impurity_decrease * node.samples
        self._accumulate_importance(node.left, importances)
        self._accumulate_importance(node.right, importances)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.root is None:
            return "DecisionTree(not fitted)"
        if self.depth > 4:
            return (
                f"DecisionTree("
                f"criterion={self.criterion!r}, "
                f"max_depth={self.max_depth}, "
                f"depth={self.depth}, "
                f"n_leaves={self.n_leaves}, "
                f"n_features={self.n_features_})"
            )
        lines: list[str] = []
        self._repr_node(self.root, lines, indent=0)
        return "\n".join(lines)

    def _repr_node(
        self, node: Optional[Node], lines: list[str], indent: int
    ) -> None:
        if node is None:
            return
        prefix = "  " * indent
        if node.is_leaf_:
            dist_str = np.array2string(node.value, precision=3, suppress_small=True)
            lines.append(
                f"{prefix}Leaf | samples={node.samples} "
                f"impurity={node.impurity:.4f} value={dist_str}"
            )
        else:
            lines.append(
                f"{prefix}Split | feature={node.feature} "
                f"threshold={node.threshold:.4f} "
                f"impurity={node.impurity:.4f} "
                f"samples={node.samples} "
                f"gain={node.impurity_decrease:.4f}"
            )
            self._repr_node(node.left, lines, indent + 1)
            self._repr_node(node.right, lines, indent + 1)


# ---------------------------------------------------------------------------
# DecisionStump (convenience subclass for AdaBoost)
# ---------------------------------------------------------------------------

class DecisionStump(DecisionTree):
    """Depth-1 decision tree — the weak learner for AdaBoost.

    Parameters
    ----------
    criterion    : {"gini", "entropy"}
    random_state : int or None
    """

    def __init__(
        self,
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)
