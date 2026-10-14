"""Hyperparameter search over leak-safe Pipelines."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocess_pipeline


SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "sklearn_logistic": {
        "model__C": np.logspace(-2, 2, 20),
        "model__solver": ["lbfgs"],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 150, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [1, 3, 5],
        "model__subsample": [0.8, 1.0],
    },
    "random_forest": {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": ["sqrt", 0.5, None],
    },
}


def _base_pipeline(
    model_name: str,
    *,
    age_strategy: str,
    random_state: int,
) -> Pipeline:
    if model_name == "sklearn_logistic":
        estimator = LogisticRegression(max_iter=2000, random_state=random_state)
        scale = True
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=random_state)
        scale = False
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        scale = False
    else:
        raise ValueError(
            f"Tuning not configured for '{model_name}'. "
            f"Choose from: {sorted(SEARCH_SPACES)}"
        )

    pre = build_preprocess_pipeline(age_strategy=age_strategy, scale_numeric=scale)
    return Pipeline([("preprocess", pre), ("model", estimator)])


def tune_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    model_name: str = "gradient_boosting",
    age_strategy: str = "knn",
    n_iter: int = 20,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Randomized search over a named model's hyperparameters.

    Returns the fitted search object summary: best params, best CV score,
    and the refit best estimator (full Pipeline).
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Options: {sorted(SEARCH_SPACES)}"
        )

    pipe = _base_pipeline(
        model_name, age_strategy=age_strategy, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    return {
        "model_name": model_name,
        "scoring": scoring,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
        "cv_results": results[
            ["mean_test_score", "std_test_score", "params"]
        ].reset_index(drop=True),
        "search": search,
    }
