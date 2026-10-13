"""Model zoo and evaluation helpers."""

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
from sklearn.preprocessing import StandardScaler

from .custom_logistic import CustomLogisticRegression


def build_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """Return a curated set of classifiers spanning linear and tree families."""
    return {
        "custom_logistic": CustomLogisticRegression(
            learning_rate=0.5,
            max_iter=4000,
            standardize=True,
            random_state=random_state,
        ),
        "sklearn_logistic": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def evaluate_models(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Stratified k-fold comparison across the model zoo.

    Returns a leaderboard DataFrame and raw fold metrics for each model.
    """
    models = build_model_zoo(random_state=random_state)
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
    }
    if proba is not None:
        report["roc_auc"] = float(roc_auc_score(y_test, proba))
    return report
