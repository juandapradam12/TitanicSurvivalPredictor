"""Probability calibration and decision-threshold optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


def calibrate_model(
    model,
    X,
    y,
    *,
    method: str = "isotonic",
    cv_splits: int = 3,
    random_state: int = 42,
):
    """
    Wrap a classifier with ``CalibratedClassifierCV``.

    The base model should already be a full preprocess → estimator Pipeline.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    calibrated.fit(X, y)
    return calibrated


def find_best_threshold(
    y_true,
    y_proba,
    *,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sweep probability thresholds and pick the one maximizing ``metric``."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        row = {
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"Unsupported metric '{metric}'. Use accuracy or f1.")
    best_idx = int(table[metric].idxmax())
    best = table.loc[best_idx]
    return {
        "metric": metric,
        "best_threshold": float(best["threshold"]),
        "best_score": float(best[metric]),
        "curve": table,
    }


def evaluate_probabilities(
    model,
    X_test,
    y_test,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Holdout metrics at a chosen probability threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "y_proba": proba,
        "y_pred": preds,
    }


def calibration_report(
    model,
    X,
    y,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    method: str = "isotonic",
    calibrate: bool = True,
) -> Dict[str, Any]:
    """
    Train/holdout split → optional calibration → threshold sweep → metrics.

    Returns uncalibrated and (optionally) calibrated probability diagnostics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    raw = model
    # Clone-ish: fit a fresh copy if unfitted; callers usually pass unfitted Pipeline.
    from sklearn.base import clone

    raw = clone(model)
    raw.fit(X_train, y_train)
    raw_eval = evaluate_probabilities(raw, X_test, y_test, threshold=0.5)
    raw_thresh = find_best_threshold(y_test, raw_eval["y_proba"], metric="f1")
    raw_tuned = evaluate_probabilities(
        raw, X_test, y_test, threshold=raw_thresh["best_threshold"]
    )

    result: Dict[str, Any] = {
        "X_test": X_test,
        "y_test": y_test,
        "raw_model": raw,
        "raw_at_0_5": {k: v for k, v in raw_eval.items() if k not in {"y_proba", "y_pred"}},
        "raw_best_threshold": raw_thresh,
        "raw_at_best_threshold": {
            k: v for k, v in raw_tuned.items() if k not in {"y_proba", "y_pred"}
        },
        "raw_proba": raw_eval["y_proba"],
    }

    if calibrate:
        calibrated = calibrate_model(
            clone(model),
            X_train,
            y_train,
            method=method,
            random_state=random_state,
        )
        cal_eval = evaluate_probabilities(calibrated, X_test, y_test, threshold=0.5)
        cal_thresh = find_best_threshold(y_test, cal_eval["y_proba"], metric="f1")
        cal_tuned = evaluate_probabilities(
            calibrated, X_test, y_test, threshold=cal_thresh["best_threshold"]
        )
        result.update(
            {
                "calibrated_model": calibrated,
                "calibrated_at_0_5": {
                    k: v for k, v in cal_eval.items() if k not in {"y_proba", "y_pred"}
                },
                "calibrated_best_threshold": cal_thresh,
                "calibrated_at_best_threshold": {
                    k: v for k, v in cal_tuned.items() if k not in {"y_proba", "y_pred"}
                },
                "calibrated_proba": cal_eval["y_proba"],
                "calibration_method": method,
            }
        )

    return result


def plot_reliability(
    y_true,
    proba_raw,
    proba_calibrated: Optional[np.ndarray] = None,
    *,
    n_bins: int = 8,
    output_path: Optional[Path | str] = None,
    title: str = "Reliability curve",
):
    """Plot calibration (reliability) curves before/after calibration."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfectly calibrated")

    frac_raw, mean_raw = calibration_curve(
        y_true, proba_raw, n_bins=n_bins, strategy="quantile"
    )
    ax.plot(mean_raw, frac_raw, "o-", label="Uncalibrated", color="#c44e52")

    if proba_calibrated is not None:
        frac_cal, mean_cal = calibration_curve(
            y_true, proba_calibrated, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_cal, frac_cal, "o-", label="Calibrated", color="#2c6eaa")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
    return fig


def plot_roc_pr(
    y_true,
    y_proba,
    *,
    output_path: Optional[Path | str] = None,
    title_prefix: str = "Holdout",
) -> Tuple[Any, Any]:
    """Save side-by-side ROC and Precision-Recall curves."""
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#2c6eaa", label=f"ROC-AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#888")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{title_prefix} ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="#c44e52")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} Precision-Recall")
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
    return fig, auc
