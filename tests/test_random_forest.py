import numpy as np
import pytest
from src.bagging.random_forest import RandomForestClassifier


def test_random_forest_fit_predict():
    # Simple linearly separable 2D dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_random_forest_oob_score():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = RandomForestClassifier(n_estimators=30, bootstrap=True, oob_score=True, random_state=42)
    clf.fit(X, y)

    oob = clf.oob_score_
    assert oob >= 0.0 and oob <= 1.0


def test_random_forest_parallel():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    # Run with 2 jobs and check that prediction is correct and fit works
    clf = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_random_forest_reproducibility():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf1 = RandomForestClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)

    clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
    clf2.fit(X, y)

    assert np.array_equal(clf1.predict_proba(X), clf2.predict_proba(X))


def test_random_forest_edge_cases():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    # OOB score requires bootstrap=True
    with pytest.raises(ValueError):
        RandomForestClassifier(oob_score=True, bootstrap=False).fit(X, y)

    # Accessing oob_score_ without oob_score=True
    clf = RandomForestClassifier(oob_score=False).fit(X, y)
    with pytest.raises(AttributeError):
        _ = clf.oob_score_

    # Accessing oob_score_ before fit
    clf_unfitted = RandomForestClassifier(oob_score=True)
    with pytest.raises(AttributeError):
        _ = clf_unfitted.oob_score_


def test_random_forest_feature_importances():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    importances = clf.feature_importances_
    assert importances.shape == (2,)
    assert np.isclose(importances.sum(), 1.0)
