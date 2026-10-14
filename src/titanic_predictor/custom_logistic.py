"""
From-scratch logistic regression via batch gradient ascent on the log-likelihood.

Educational implementation kept close to the original notebook math, but
vectorized for speed and wrapped as a scikit-learn-style estimator so it can
enter the same evaluation harness as library models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _as_pm1(y: np.ndarray) -> np.ndarray:
    """Map {0,1} / {-1,1} labels to {-1, 1}."""
    y = np.asarray(y).reshape(-1)
    unique = set(np.unique(y).tolist())
    if unique <= {-1, 1}:
        return y.astype(float)
    if unique <= {0, 1}:
        return np.where(y == 1, 1.0, -1.0)
    raise ValueError(f"Unsupported label set: {unique}")


@dataclass
class FitHistory:
    iterations: int
    final_norm: float


class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Binary logistic regression trained by gradient ascent.

    Likelihood (labels in {-1, 1}):

        L(w) = Σ log σ(y_i x_iᵀ w)

    Gradient:

        ∇L = Σ (1 − σ(y_i x_iᵀ w)) y_i x_i

    where σ(z) = 1 / (1 + e^{-z}).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 5000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.random_state = random_state

    def _design(self, X: np.ndarray) -> np.ndarray:
        if self.standardize:
            X = (X - self.mean_) / self.scale_
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.concatenate([ones, X], axis=1)
        return X

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable logistic sigmoid.
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        y_pm1 = _as_pm1(y)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1])
        self.scale_ = X.std(axis=0) if self.standardize else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0

        X_des = self._design(X)
        w = np.zeros(X_des.shape[1], dtype=float)

        for iteration in range(1, self.max_iter + 1):
            margin = y_pm1 * (X_des @ w)
            sigma = self._sigmoid(margin)
            grad = ((1.0 - sigma) * y_pm1) @ X_des
            step = self.learning_rate * grad / X_des.shape[0]
            w = w + step
            if np.linalg.norm(step) < self.tol:
                break

        self.coef_ = w[1:][None, :] if self.fit_intercept else w[None, :]
        self.intercept_ = np.array([w[0]]) if self.fit_intercept else np.array([0.0])
        self.weight_vector_ = w
        self.history_ = FitHistory(iterations=iteration, final_norm=float(np.linalg.norm(step)))
        return self

    def decision_function(self, X):
        check_is_fitted(self, "weight_vector_")
        X = check_array(X, dtype=float)
        return self._design(X) @ self.weight_vector_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        pos = self._sigmoid(scores)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
