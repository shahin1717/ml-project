import numpy as np
import pytest
from src.trees.decision_tree import DecisionTree, DecisionStump, gini, entropy


def test_gini_entropy_functions():
    # Test Gini and Entropy metrics directly
    y = np.array([0, 0, 1, 1])
    assert np.isclose(gini(y), 0.5)
    assert np.isclose(entropy(y), 1.0)

    y_pure = np.array([1, 1, 1])
    assert np.isclose(gini(y_pure), 0.0)
    assert np.isclose(entropy(y_pure), 0.0)

    # Empty/zero weight case
    assert np.isclose(gini(np.array([]), np.array([])), 0.0)
    assert np.isclose(entropy(np.array([]), np.array([])), 0.0)


def test_decision_tree_fit_predict():
    # Simple linearly separable 2D dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = DecisionTree(max_depth=2, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_decision_tree_entropy_criterion():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = DecisionTree(criterion="entropy", max_depth=2, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_decision_tree_properties():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = DecisionTree(max_depth=3, random_state=42)
    clf.fit(X, y)

    assert clf.depth >= 1
    assert clf.n_leaves >= 2
    
    importances = clf.feature_importances()
    assert importances.shape == (2,)
    assert np.isclose(importances.sum(), 1.0)


def test_decision_tree_edge_cases():
    # 1. Pure node
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    clf = DecisionTree().fit(X, y)
    assert clf.depth == 0
    assert clf.predict(X)[0] == 1

    # 2. Identical feature vectors with different labels
    X_identical = np.array([[1.0, 1.0], [1.0, 1.0]])
    y_diff = np.array([0, 1])
    clf = DecisionTree().fit(X_identical, y_diff)
    assert clf.depth == 0

    # 3. Invalid inputs
    clf = DecisionTree()
    with pytest.raises(ValueError):
        clf.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))  # not 2D
    with pytest.raises(ValueError):
        clf.fit(np.array([[1], [2]]), np.array([1, 2, 3]))  # len mismatch
    with pytest.raises(ValueError):
        clf.fit(np.array([[1], [2]]), np.array([1, 2]), sample_weight=np.array([1]))  # weight len mismatch


def test_decision_tree_representation():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    # Unfitted repr
    clf_unfitted = DecisionTree()
    assert "not fitted" in repr(clf_unfitted)

    # Small depth fitted repr
    clf_small = DecisionTree(max_depth=2).fit(X, y)
    repr_str = repr(clf_small)
    assert "Split" in repr_str or "Leaf" in repr_str

    # Large depth fitted repr (brief summary)
    X_deep = np.arange(12).reshape(-1, 1)
    y_deep = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf_large = DecisionTree(max_depth=10).fit(X_deep, y_deep)
    assert clf_large.depth > 4
    assert "n_leaves" in repr(clf_large)


def test_decision_stump():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    stump = DecisionStump()
    stump.fit(X, y)
    assert stump.max_depth == 1
    assert stump.depth <= 1
