"""AdaBoost classifier (discrete SAMME) implementation.

Implements discrete SAMME variant using DecisionStump weak learners.
Follows requirements in §2.2 of ML_FINAL_PROJECT.tex.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import numpy as np

from src.trees.decision_tree import DecisionStump


class AdaBoostClassifier:
    """AdaBoost classifier using DecisionStump weak learners.

    Parameters
    ----------
    n_estimators : int, default=50
        Maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    algorithm : {"SAMME", "SAMME.R"}, default="SAMME"
        The boosting algorithm to use. "SAMME" implements the discrete
        SAMME variant, and "SAMME.R" implements the real SAMME.R variant.
    criterion : {"gini", "entropy"}, default="gini"
        The impurity criterion to use for the decision stumps.
    random_state : int or None, default=None
        Controls the random seed given to each estimator at each boosting round.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.criterion = criterion
        self.random_state = random_state

        if algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError(f"algorithm must be 'SAMME' or 'SAMME.R', got {algorithm}")

        self.estimators_: list[DecisionStump] = []
        self.estimator_weights_: list[float] = []
        self.estimator_errors_: list[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the AdaBoost classifier.

        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training vector.
        y : shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} entries."
            )

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(f"Number of classes must be greater than 1, got {self.n_classes_}.")

        n_samples = X.shape[0]
        w = np.ones(n_samples, dtype=float) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for m in range(self.n_estimators):
            # Seed a separate generator per round based on random_state + m
            stump_seed = (
                None if self.random_state is None else self.random_state + m
            )
            stump = DecisionStump(
                criterion=self.criterion, random_state=stump_seed
            )
            stump.fit(X, y, sample_weight=w)

            w_sum = w.sum()
            if w_sum == 0:
                # If weights sum to 0, stop boosting
                break

            if self.algorithm == "SAMME.R":
                proba = stump.predict_proba(X)
                # Align probabilities to the global classes
                proba = _align_proba(stump, X, self.classes_)

                # Predict classes based on probabilities
                preds = self.classes_[np.argmax(proba, axis=1)]
                incorrect = preds != y

                # Compute weighted error
                err_m = float(w[incorrect].sum() / w_sum)

                # Stop boosting if perfect fit
                if err_m <= 0:
                    self.estimators_.append(stump)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break

                # Target coding: y_k = 1 if y == class_k else -1 / (K - 1)
                y_codes = np.array([-1.0 / (self.n_classes_ - 1), 1.0])
                y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])

                # Clip probabilities to avoid log of 0
                np.clip(proba, 1e-15, None, out=proba)

                # Update weights using multiclass AdaBoost SAMME.R formula
                factor = (self.n_classes_ - 1.0) / self.n_classes_
                estimator_weight = (
                    -1.0
                    * self.learning_rate
                    * factor
                    * (y_coding * np.log(proba)).sum(axis=1)
                )

                # Only boost positive weights
                w *= np.exp(estimator_weight * (w > 0))
                # Renormalize weights
                w /= w.sum()

                self.estimators_.append(stump)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(err_m)

            else:  # SAMME
                preds = stump.predict(X)
                incorrect = preds != y

                # Compute weighted error
                err_m = float(w[incorrect].sum() / w_sum)

                # Clip to avoid division by zero or negative log
                if err_m == 0:
                    err_m = 1e-10

                # Early stopping if stump is worse than random guessing.
                # For multiclass SAMME, random guess accuracy is 1/K, so stop if err >= (K-1)/K.
                # For binary K=2, this is 0.5.
                random_guess_err = (self.n_classes_ - 1) / self.n_classes_
                if err_m >= random_guess_err:
                    break

                # SAMME weight formula: alpha = ln((1-err)/err) + ln(K-1)
                # Scaled by learning_rate
                alpha_m = self.learning_rate * (
                    math.log((1.0 - err_m) / err_m)
                    + math.log(self.n_classes_ - 1 if self.n_classes_ > 1 else 1.0)
                )

                # Update weights: w *= exp(alpha * I(y != y_pred))
                # Only scale up misclassified samples
                w *= np.exp(alpha_m * incorrect)
                # Normalize weights
                w /= w.sum()

                self.estimators_.append(stump)
                self.estimator_weights_.append(alpha_m)
                self.estimator_errors_.append(err_m)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for X.

        Parameters
        ----------
        X : shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : shape (n_samples,)
            Predicted classes.
        """
        if not self.estimators_:
            raise ValueError("AdaBoostClassifier is not fitted yet.")

        # Compute raw weighted class scores
        scores = self._compute_scores(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X : shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        p : shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if not self.estimators_:
            raise ValueError("AdaBoostClassifier is not fitted yet.")

        scores = self._compute_scores(X)
        # Shift scores for numerical stability in exponentiation
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute the weighted majority voting scores for each class."""
        X = np.asarray(X)
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_), dtype=float)

        assert self.classes_ is not None
        if self.algorithm == "SAMME.R":
            for stump in self.estimators_:
                proba = _align_proba(stump, X, self.classes_)
                np.clip(proba, 1e-15, None, out=proba)
                log_proba = np.log(proba)
                centered = log_proba - log_proba.mean(axis=1, keepdims=True)
                scores += centered
            scores /= len(self.estimators_)
        else:  # SAMME
            for stump, alpha in zip(self.estimators_, self.estimator_weights_):
                preds = stump.predict(X)
                # Accumulate weight for predicted class
                for k, c in enumerate(self.classes_):
                    scores[preds == c, k] += alpha

        return scores

    @property
    def estimator_weights(self) -> np.ndarray:
        """The weights for each estimator in the ensemble."""
        return np.array(self.estimator_weights_, dtype=float)

    @property
    def estimator_errors(self) -> np.ndarray:
        """The classification error for each estimator in the ensemble."""
        return np.array(self.estimator_errors_, dtype=float)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Return predictions staged after each boosting round.

        Parameters
        ----------
        X : shape (n_samples, n_features)
            Input samples.

        Yields
        ------
        y : shape (n_samples,)
            Staged class predictions.
        """
        if not self.estimators_:
            raise ValueError("AdaBoostClassifier is not fitted yet.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_), dtype=float)

        assert self.classes_ is not None
        if self.algorithm == "SAMME.R":
            for stump in self.estimators_:
                proba = _align_proba(stump, X, self.classes_)
                np.clip(proba, 1e-15, None, out=proba)
                log_proba = np.log(proba)
                centered = log_proba - log_proba.mean(axis=1, keepdims=True)
                scores += centered
                yield self.classes_[np.argmax(scores, axis=1)]
        else:  # SAMME
            for stump, alpha in zip(self.estimators_, self.estimator_weights_):
                preds = stump.predict(X)
                for k, c in enumerate(self.classes_):
                    scores[preds == c, k] += alpha
                yield self.classes_[np.argmax(scores, axis=1)]

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Return probability estimates staged after each boosting round.

        Parameters
        ----------
        X : shape (n_samples, n_features)
            Input samples.

        Yields
        ------
        p : shape (n_samples, n_classes)
            Staged class probability estimates.
        """
        if not self.estimators_:
            raise ValueError("AdaBoostClassifier is not fitted yet.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_), dtype=float)

        assert self.classes_ is not None
        if self.algorithm == "SAMME.R":
            for m, stump in enumerate(self.estimators_):
                proba = _align_proba(stump, X, self.classes_)
                np.clip(proba, 1e-15, None, out=proba)
                log_proba = np.log(proba)
                centered = log_proba - log_proba.mean(axis=1, keepdims=True)
                scores += centered
                curr_scores = scores / (m + 1)
                scores_shifted = curr_scores - np.max(curr_scores, axis=1, keepdims=True)
                exp_scores = np.exp(scores_shifted)
                yield exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:  # SAMME
            for stump, alpha in zip(self.estimators_, self.estimator_weights_):
                preds = stump.predict(X)
                for k, c in enumerate(self.classes_):
                    scores[preds == c, k] += alpha
                scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
                exp_scores = np.exp(scores_shifted)
                yield exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _align_proba(estimator: DecisionStump, X: np.ndarray, global_classes: np.ndarray) -> np.ndarray:
    """Align predicted probabilities of the estimator with global classes.

    In case the estimator did not see all classes during fit.
    """
    proba = estimator.predict_proba(X)
    assert estimator.classes_ is not None
    if np.array_equal(estimator.classes_, global_classes):
        return proba

    aligned_proba = np.zeros((X.shape[0], len(global_classes)), dtype=proba.dtype)
    class_to_idx = {c: i for i, c in enumerate(global_classes)}
    for i, c in enumerate(estimator.classes_):
        if c in class_to_idx:
            aligned_proba[:, class_to_idx[c]] = proba[:, i]
    return aligned_proba
