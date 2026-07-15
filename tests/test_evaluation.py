from __future__ import annotations
import numpy as np
import pytest
from src.metrics.evaluation import (
    accuracy_score,
    confusion_matrix,
    precision_recall_f1,
)


def test_accuracy_score() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    assert accuracy_score(y_true, y_pred) == pytest.approx(5 / 6)

    # Empty array
    assert accuracy_score(np.array([]), np.array([])) == 0.0

    # Shape mismatch
    with pytest.raises(ValueError):
        accuracy_score(np.array([1]), np.array([1, 2]))


def test_confusion_matrix() -> None:
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    cm = confusion_matrix(y_true, y_pred)
    
    # Classes are {0, 1}
    # True 0, Pred 0 (TN) = 2
    # True 0, Pred 1 (FP) = 0
    # True 1, Pred 0 (FN) = 1
    # True 1, Pred 1 (TP) = 2
    expected = np.array([[2, 0], [1, 2]])
    assert np.array_equal(cm, expected)

    # Empty array
    assert np.array_equal(confusion_matrix(np.array([]), np.array([])), np.zeros((0, 0)))


def test_precision_recall_f1() -> None:
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    
    # Class 0:
    # TP=2, FP=1, FN=0 -> Precision = 2/3, Recall = 1.0, F1 = 4/5
    # Class 1:
    # TP=2, FP=0, FN=1 -> Precision = 1.0, Recall = 2/3, F1 = 4/5
    
    # Macro average:
    # Precision = (2/3 + 1.0) / 2 = 5/6 = 0.8333
    # Recall = (1.0 + 2/3) / 2 = 5/6 = 0.8333
    # F1 = (4/5 + 4/5) / 2 = 0.80
    prec, rec, f1 = precision_recall_f1(y_true, y_pred, average="macro")
    assert prec == pytest.approx(5 / 6)
    assert rec == pytest.approx(5 / 6)
    assert f1 == pytest.approx(0.8)

    # Binary average (class 1)
    prec_b, rec_b, f1_b = precision_recall_f1(y_true, y_pred, average="binary")
    assert prec_b == pytest.approx(1.0)
    assert rec_b == pytest.approx(2 / 3)
    assert f1_b == pytest.approx(0.8)

    # Empty array
    assert precision_recall_f1(np.array([]), np.array([])) == (0.0, 0.0, 0.0)
