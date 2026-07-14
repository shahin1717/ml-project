from __future__ import annotations
import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the accuracy score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    accuracy : float
        Fraction of correctly classified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    C : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where C[i, j] is the number of observations known to be
        in group i and predicted to be in group j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    if n_labels == 0:
        return np.zeros((0, 0), dtype=int)

    label_to_ind = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_ind[t], label_to_ind[p]] += 1
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1-score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    average : {"macro", "binary"}, default="macro"
        If "macro", compute metrics for each label and find their unweighted mean.
        If "binary", only report results for class 1 (only valid for binary data).

    Returns
    -------
    precision : float
    recall : float
    f1_score : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(labels) == 0:
        return 0.0, 0.0, 0.0

    precisions = []
    recalls = []
    f1s = []

    # Compute metrics for each class
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    if average == "binary":
        # Return metrics for class 1 if present, otherwise classes[0]
        ind_c1 = np.where(labels == 1)[0]
        if len(ind_c1) > 0:
            idx = ind_c1[0]
        else:
            idx = 0
        return precisions[idx], recalls[idx], f1s[idx]

    # Default to macro average
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))
