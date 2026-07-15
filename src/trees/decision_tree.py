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
    parent_impurity: Optional[float] = None,
) -> tuple[Optional[int], Optional[float], float]:
    """Find the best (feature, threshold, gain) using a vectorized cumsum sweep.

    For each candidate feature the samples are sorted once; cumulative weighted
    class counts are computed via ``np.cumsum`` so the entire sweep over all
    N-1 cut points is done with numpy C-loops instead of a Python for-loop.
    Complexity: O(N log N) per feature (sort) + O(N · n_classes) (cumsum).

    Thresholds are midpoints between consecutive distinct feature values (DT.3).

    Returns
    -------
    (best_feature, best_threshold, best_gain)
    best_feature is None when no valid split exists.
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
    n_classes = len(classes)
    # Map labels to 0..n_classes-1 for fast indexing
    class_idx = np.searchsorted(classes, y)   # shape (n_samples,)

    total_w = sample_weight.sum()
    if parent_impurity is None:
        parent_impurity = _impurity_fn(criterion)(y, sample_weight)

    best_gain: float = -1.0
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None

    for feature in feature_indices:
        order = np.argsort(X[:, feature], kind="mergesort")
        xs   = X[order, feature]          # sorted feature values
        cidx = class_idx[order]           # sorted class indices
        ws   = sample_weight[order]       # sorted weights

        # Weighted one-hot: shape (n_samples, n_classes)
        w_onehot = np.zeros((n_samples, n_classes), dtype=float)
        w_onehot[np.arange(n_samples), cidx] = ws

        # left_cum[i] = weighted class counts for samples 0..i (inclusive)
        left_cum = np.cumsum(w_onehot, axis=0)          # (n_samples, n_classes)
        left_w   = np.cumsum(ws)                         # (n_samples,)
        right_cum = left_cum[-1] - left_cum              # (n_samples, n_classes)
        right_w   = total_w - left_w                     # (n_samples,)

        # Only evaluate thresholds between *distinct* consecutive values.
        # valid[i] == True means xs[i] < xs[i+1], so a threshold exists between them.
        valid = xs[:-1] != xs[1:]                        # (n_samples-1,)
        if not valid.any():
            continue

        # Slice to candidate cut-points (indices 0..n_samples-2)
        lc = left_cum[:-1][valid]    # (n_valid, n_classes)
        lw = left_w[:-1][valid]      # (n_valid,)
        rc = right_cum[:-1][valid]   # (n_valid, n_classes)
        rw = right_w[:-1][valid]     # (n_valid,)

        # Compute impurity at each cut-point vectorially
        if criterion == "gini":
            lp = lc / lw[:, None]                        # left class probs
            rp = rc / rw[:, None]                        # right class probs
            left_imp  = 1.0 - (lp ** 2).sum(axis=1)     # (n_valid,)
            right_imp = 1.0 - (rp ** 2).sum(axis=1)
        else:  # entropy
            eps = 1e-12
            lp = lc / lw[:, None]
            rp = rc / rw[:, None]
            left_imp  = -(lp * np.log2(lp + eps)).sum(axis=1)
            right_imp = -(rp * np.log2(rp + eps)).sum(axis=1)

        gains = parent_impurity - (lw / total_w) * left_imp - (rw / total_w) * right_imp

        best_i = int(np.argmax(gains))
        if gains[best_i] > best_gain:
            best_gain = float(gains[best_i])
            best_feature = int(feature)
            # threshold = midpoint between the two original sorted values
            valid_idx = np.where(valid)[0]
            i = valid_idx[best_i]
            best_threshold = float((xs[i] + xs[i + 1]) / 2.0)

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
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} entries."
            )
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != len(y):
                raise ValueError(
                    f"sample_weight has {sample_weight.shape[0]} entries but y has {len(y)}."
                )

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
        assert self.classes_ is not None
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
        if np.all(X == X[0]):
            return make_leaf()

        # --- Best split --- (pass pre-computed impurity to avoid recomputation)
        feature, threshold, gain = best_split(
            X, y, sample_weight,
            criterion=self.criterion,
            max_features=self.max_features,
            rng=self.rng,
            parent_impurity=node_impurity,
        )

        # Accept zero-gain splits (matches sklearn's min_impurity_decrease=0 default).
        # This resolves plateau-root cases like XOR where the first split has gain==0
        # but depth-2 children are pure.  The only hard stop is feature is None,
        # which means best_split found no valid candidate at all.
        if feature is None:
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
        assert self.root is not None
        return np.array([self._traverse_proba(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def _traverse_proba(self, x: np.ndarray, node: Optional[Node]) -> np.ndarray:
        assert node is not None
        if node.is_leaf_:
            assert node.value is not None
            return node.value
        assert node.feature is not None
        assert node.threshold is not None
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
        assert node.value is not None
        if node.is_leaf_:
            dist_str = np.array2string(node.value, precision=3, suppress_small=True)
            lines.append(
                f"{prefix}Leaf | samples={node.samples} "
                f"impurity={node.impurity:.4f} value={dist_str}"
            )
        else:
            dist_str = np.array2string(node.value, precision=3, suppress_small=True)
            lines.append(
                f"{prefix}Split | feature={node.feature} "
                f"threshold={node.threshold:.4f} "
                f"impurity={node.impurity:.4f} "
                f"samples={node.samples} "
                f"gain={node.impurity_decrease:.4f} "
                f"value={dist_str}"
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
