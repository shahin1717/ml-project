import numpy as np
import pytest
from src.boosting.adaboost import AdaBoostClassifier


def test_adaboost_fit_predict():
    # Linear separable classification dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = AdaBoostClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_adaboost_unfitted():
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError):
        clf.predict(np.array([[1, 2]]))
    with pytest.raises(ValueError):
        clf.predict_proba(np.array([[1, 2]]))
    with pytest.raises(ValueError):
        list(clf.staged_predict(np.array([[1, 2]])))


def test_adaboost_staged_predict():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    staged_preds = list(clf.staged_predict(X))
    assert len(staged_preds) > 0
    assert staged_preds[-1].shape == (6,)


def test_adaboost_early_stopping():
    # Test early stopping when stump error >= random guess error.
    # On identical features with balanced labels, the best stump gets error 0.5
    # which is >= random_guess_err (0.5 for K=2). It should stop after the first round.
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1])

    clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    assert len(clf.estimators_) == 0


def test_adaboost_multiclass():
    # Simple 3-class dataset
    X = np.array([[1], [2], [10], [11], [20], [21]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])

    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (6,)
    
    # We should be able to get valid class probabilities
    probs = clf.predict_proba(X)
    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
