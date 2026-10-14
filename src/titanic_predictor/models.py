"""Model zoo and evaluation helpers (leak-safe Pipelines)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .custom_logistic import CustomLogisticRegression
from .preprocessing import build_preprocess_pipeline


def build_model_zoo(
    random_state: int = 42,
    *,
    age_strategy: str = "knn",
) -> Dict[str, Pipeline]:
    """
    Return classifiers wrapped in full preprocess → model Pipelines.

    Preprocessing (feature build, age imputation, encoding, scaling) is fitted
    inside each CV fold via these Pipelines — no global imputation leakage.
    """
    def _pipe(estimator, *, scale_numeric: bool) -> Pipeline:
        pre = build_preprocess_pipeline(
            age_strategy=age_strategy,
            scale_numeric=scale_numeric,
        )
        return Pipeline([("preprocess", pre), ("model", estimator)])

    return {
        "custom_logistic": _pipe(
            CustomLogisticRegression(
                learning_rate=0.5,
                max_iter=4000,
                # Features are already scaled in preprocess for linear models.
                standardize=False,
                random_state=random_state,
            ),
            scale_numeric=True,
        ),
        "sklearn_logistic": _pipe(
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=random_state,
            ),
            scale_numeric=True,
        ),
        "random_forest": _pipe(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "gradient_boosting": _pipe(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    cv_splits: int = 5,
    random_state: int = 42,
    age_strategy: str = "knn",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Stratified k-fold comparison across the leak-safe model zoo.

    Returns a leaderboard DataFrame and raw fold metrics for each model.
    """
    models = build_model_zoo(random_state=random_state, age_strategy=age_strategy)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        cv_result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=None,
            return_train_score=False,
        )
        row = {
            "model": name,
            "accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
            "accuracy_std": float(np.std(cv_result["test_accuracy"])),
            "f1_mean": float(np.mean(cv_result["test_f1"])),
            "f1_std": float(np.std(cv_result["test_f1"])),
            "roc_auc_mean": float(np.mean(cv_result["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_result["test_roc_auc"])),
            "fit_time_mean": float(np.mean(cv_result["fit_time"])),
        }
        rows.append(row)
        details[name] = {k: v.tolist() for k, v in cv_result.items()}

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard, details


def holdout_report(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Dict[str, Any]:
    """Fit on a holdout split and return common classification metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]

    report = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds, digits=3),
        "fitted_model": model,
    }
    if proba is not None:
        report["roc_auc"] = float(roc_auc_score(y_test, proba))
    return report
