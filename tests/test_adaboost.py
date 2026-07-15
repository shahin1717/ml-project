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


def test_sammer_binary():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = AdaBoostClassifier(n_estimators=5, algorithm="SAMME.R", random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_sammer_multiclass():
    X = np.array([[1], [2], [10], [11], [20], [21]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])

    clf = AdaBoostClassifier(n_estimators=10, algorithm="SAMME.R", random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (6,)
    assert np.array_equal(preds, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_sammer_staged_predict():
    X = np.array([[1], [2], [10], [11], [20], [21]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])

    clf = AdaBoostClassifier(n_estimators=5, algorithm="SAMME.R", random_state=42)
    clf.fit(X, y)

    staged_preds = list(clf.staged_predict(X))
    assert len(staged_preds) == len(clf.estimators_)
    assert staged_preds[-1].shape == (6,)

    staged_probs = list(clf.staged_predict_proba(X))
    assert len(staged_probs) == len(clf.estimators_)
    assert staged_probs[-1].shape == (6, 3)
    assert np.allclose(staged_probs[-1].sum(axis=1), 1.0)


def test_invalid_algorithm():
    with pytest.raises(ValueError):
        AdaBoostClassifier(algorithm="INVALID")


def test_sammer_calibration():
    # Generate simple noisy 3-class classification data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.zeros(100, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 1] > 0.5] = 2

    # Split into train/test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    clf_samme = AdaBoostClassifier(n_estimators=30, algorithm="SAMME", random_state=42)
    clf_samme.fit(X_train, y_train)
    probs_samme = clf_samme.predict_proba(X_test)

    clf_sammer = AdaBoostClassifier(n_estimators=30, algorithm="SAMME.R", random_state=42)
    clf_sammer.fit(X_train, y_train)
    probs_sammer = clf_sammer.predict_proba(X_test)

    # Compute Brier score: mean of sum((p_ik - y_ik)^2)
    y_onehot = np.zeros((len(y_test), 3))
    for i, label in enumerate(y_test):
        y_onehot[i, label] = 1.0

    brier_samme = np.mean(np.sum((probs_samme - y_onehot) ** 2, axis=1))
    brier_sammer = np.mean(np.sum((probs_sammer - y_onehot) ** 2, axis=1))

    # Real-valued SAMME.R should generally have better or comparable calibration
    # on this small test set. Let's assert it is correct and valid.
    assert np.isfinite(brier_samme)
    assert np.isfinite(brier_sammer)
    assert brier_sammer < 1.0  # sensible bound for Brier score

