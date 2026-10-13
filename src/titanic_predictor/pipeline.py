"""End-to-end training and benchmarking entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import engineer_features, prepare_model_matrix
from .imputation import build_knn_frame, knn_impute_age
from .models import build_model_zoo, evaluate_models, holdout_report


DEFAULT_DATA = Path(__file__).resolve().parents[2] / "train.csv"


def load_titanic(path: Optional[Path | str] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DEFAULT_DATA
    return pd.read_csv(path)


def build_xy(
    df: pd.DataFrame,
    *,
    use_knn_age: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct the modeling matrix used by the enhanced pipeline.

    When ``use_knn_age`` is True, Age is imputed with scaled k-NN (classic
    notebook path). Otherwise, group-median AgeFilled from feature engineering
    is used.
    """
    clean = df.loc[df["Embarked"].notna()].copy()
    y = clean["Survived"].astype(int)

    if use_knn_age:
        knn_frame = build_knn_frame(clean)
        knn_imputed = knn_impute_age(knn_frame)
        engineered = engineer_features(clean)
        engineered["AgeFilled"] = knn_imputed["Age"].values
    else:
        engineered = engineer_features(clean)

    X = prepare_model_matrix(engineered)
    return X, y


def run_benchmark(
    data_path: Optional[Path | str] = None,
    *,
    cv_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    use_knn_age: bool = True,
) -> dict:
    """
    Full benchmark: stratified CV leaderboard + holdout report for the best model.
    """
    df = load_titanic(data_path)
    X, y = build_xy(df, use_knn_age=use_knn_age)

    leaderboard, fold_details = evaluate_models(
        X, y, cv_splits=cv_splits, random_state=random_state
    )
    best_name = leaderboard.iloc[0]["model"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    best_model = build_model_zoo(random_state=random_state)[best_name]
    holdout = holdout_report(best_model, X_train, X_test, y_train, y_test)

    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "leaderboard": leaderboard,
        "fold_details": fold_details,
        "best_model": best_name,
        "holdout": holdout,
        "X": X,
        "y": y,
    }
