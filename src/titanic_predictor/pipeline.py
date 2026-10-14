"""End-to-end training, benchmarking, and submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .interpretation import (
    explain_with_shap,
    logistic_coefficient_table,
    narrative_from_coefficients,
)
from .models import build_model_zoo, evaluate_models, holdout_report
from .preprocessing import prepare_raw_xy


DEFAULT_TRAIN = Path(__file__).resolve().parents[2] / "train.csv"
DEFAULT_TEST = Path(__file__).resolve().parents[2] / "test.csv"


def load_titanic(path: Optional[Path | str] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DEFAULT_TRAIN
    return pd.read_csv(path)


def build_xy(df: pd.DataFrame, *, use_knn_age: bool = True):
    """
    Backward-compatible helper returning raw X / y for leak-safe Pipelines.

    ``use_knn_age`` is accepted for API compatibility; the age strategy is
    applied inside each model Pipeline, not here.
    """
    del use_knn_age  # strategy is selected at model-construction time
    return prepare_raw_xy(df)


def run_benchmark(
    data_path: Optional[Path | str] = None,
    *,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    age_strategy: str = "knn",
    use_knn_age: bool = True,
) -> dict:
    """
    Full benchmark: stratified CV leaderboard + holdout report for the best model.

    All preprocessing is nested inside each Pipeline so CV folds do not leak.
    """
    if not use_knn_age and age_strategy == "knn":
        age_strategy = "group_median"

    df = load_titanic(data_path)
    X, y = prepare_raw_xy(df)
    assert y is not None

    leaderboard, fold_details = evaluate_models(
        X,
        y,
        cv_splits=cv_splits,
        random_state=random_state,
        age_strategy=age_strategy,
    )
    best_name = leaderboard.iloc[0]["model"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    best_model = build_model_zoo(
        random_state=random_state, age_strategy=age_strategy
    )[best_name]
    holdout = holdout_report(best_model, X_train, X_test, y_train, y_test)
    fitted = holdout["fitted_model"]

    # Feature names after preprocess (from a transform of the train split).
    Xt_train = fitted.named_steps["preprocess"].transform(X_train)
    feature_names = list(Xt_train.columns)

    return {
        "n_samples": int(len(y)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "leaderboard": leaderboard,
        "fold_details": fold_details,
        "best_model": best_name,
        "holdout": {k: v for k, v in holdout.items() if k != "fitted_model"},
        "fitted_pipeline": fitted,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "age_strategy": age_strategy,
    }


def train_final_model(
    train_path: Optional[Path | str] = None,
    *,
    model_name: str = "gradient_boosting",
    age_strategy: str = "knn",
    random_state: int = 42,
):
    """Fit a named Pipeline on the full training set."""
    df = load_titanic(train_path)
    X, y = prepare_raw_xy(df)
    assert y is not None
    model = build_model_zoo(random_state=random_state, age_strategy=age_strategy)[
        model_name
    ]
    model.fit(X, y)
    return model, X, y


def predict_submission(
    model,
    test_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Score Kaggle test.csv and return a submission DataFrame."""
    path = Path(test_path) if test_path is not None else DEFAULT_TEST
    test_df = pd.read_csv(path)
    X_test, _ = prepare_raw_xy(test_df)
    preds = model.predict(X_test)
    return pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"].astype(int),
            "Survived": preds.astype(int),
        }
    )


def save_model(model, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str):
    return joblib.load(path)


def interpret_model(
    model,
    X: pd.DataFrame,
    *,
    max_shap_samples: int = 200,
    random_state: int = 42,
) -> dict:
    """Coefficient table (if linear) + SHAP importance for a fitted Pipeline."""
    Xt = model.named_steps["preprocess"].transform(X)
    feature_names = list(Xt.columns)
    result: dict = {"feature_names": feature_names}

    final = model.named_steps["model"]
    if hasattr(final, "coef_"):
        coef_table = logistic_coefficient_table(model, feature_names=feature_names)
        result["coefficients"] = coef_table
        result["coefficient_narrative"] = narrative_from_coefficients(coef_table)

    shap_bundle = explain_with_shap(
        model,
        X,
        max_samples=max_shap_samples,
        random_state=random_state,
    )
    result["shap_importance"] = shap_bundle["importance"]
    result["shap_values"] = shap_bundle["shap_values"]
    result["shap_explainer_type"] = shap_bundle["explainer_type"]
    result["X_shap_sample"] = shap_bundle["X_sample"]
    return result
