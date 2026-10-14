"""Age imputation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def knn_impute_age(
    features: pd.DataFrame,
    *,
    age_col: str = "Age",
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Impute missing ages with k-NN regression on scaled numeric predictors.

    Mirrors the original notebook approach, with scaling so distance is not
    dominated by Fare magnitude.
    """
    out = features.copy()
    known = out[out[age_col].notna()]
    missing = out[out[age_col].isna()]

    if missing.empty:
        return out

    predictors = [c for c in out.columns if c != age_col]
    x_train = known[predictors].astype(float)
    y_train = known[age_col].astype(float)
    x_missing = missing[predictors].astype(float)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_missing_s = scaler.transform(x_missing)

    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(x_train_s, y_train)
    preds = model.predict(x_missing_s)

    out.loc[missing.index, age_col] = preds
    # Keep ages in a plausible passenger range.
    out[age_col] = out[age_col].clip(lower=0.0, upper=80.0)
    return out


def build_knn_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal numeric frame used by the classic KNN age imputation path."""
    frame = df.copy()
    if "Embarked" in frame.columns:
        frame = frame.loc[frame["Embarked"].notna()].copy()

    drop_cols = [
        c
        for c in ["Ticket", "Cabin", "PassengerId", "Name", "Survived"]
        if c in frame.columns
    ]
    frame = frame.drop(columns=drop_cols)
    frame = pd.get_dummies(
        frame, columns=["Sex", "Embarked"], drop_first=True
    )
    return frame.astype(float)
