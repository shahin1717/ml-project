# Evaluation metrics – placeholder stub
import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)