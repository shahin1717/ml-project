"""AdaBoost classifier (discrete SAMME) implementation.

Implements discrete SAMME variant using DecisionStump weak learners.
Follows requirements in §2.2 of ML_FINAL_PROJECT.tex.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Union

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
    criterion : {"gini", "entropy"}, default="gini"
        The impurity criterion to use for the decision stumps.
    random_state : int or None, default=None
        Controls the random seed given to each estimator at each boosting round.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

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

            preds = stump.predict(X)
            incorrect = preds != y

            # Compute weighted error
            w_sum = w.sum()
            if w_sum == 0:
                # If weights sum to 0, stop boosting
                break
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
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        Uses the exponential softmax of weighted votes approach.

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

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = stump.predict(X)
            for k, c in enumerate(self.classes_):
                scores[preds == c, k] += alpha
            yield self.classes_[np.argmax(scores, axis=1)]
