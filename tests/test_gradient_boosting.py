import numpy as np
from src.boosting.gradient_boosting import GradientBoostingClassifier


def test_gbm_fit_predict():
    # Linear separable classification dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = GradientBoostingClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_gbm_reproducible():
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, size=20)

    clf1 = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)

    clf2 = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf2.fit(X, y)

    assert np.allclose(clf1.predict_proba(X), clf2.predict_proba(X))


def test_gbm_more_estimators_helps():
    # Toy problem
    X = np.random.randn(100, 5)
    # y depends nonlinearly
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    clf_weak = GradientBoostingClassifier(n_estimators=2, learning_rate=0.1, random_state=42)
    clf_weak.fit(X, y)

    clf_strong = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
    clf_strong.fit(X, y)

    preds_weak = clf_weak.predict(X)
    preds_strong = clf_strong.predict(X)

    acc_weak = np.mean(preds_weak == y)
    acc_strong = np.mean(preds_strong == y)

    # Strong model should perform at least as well as the weak one on the training set
    assert acc_strong >= acc_weak
