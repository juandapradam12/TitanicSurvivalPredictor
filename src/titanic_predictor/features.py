"""Feature engineering for the Titanic survival prediction problem."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Ms": "Miss",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Capt": "Rare",
    "Sir": "Rare",
    "Lady": "Rare",
    "Countess": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Jonkheer": "Rare",
}


def extract_title(name: str) -> str:
    """Pull the honorific from a passenger name and normalize rare titles."""
    match = re.search(r",\s*([^.]+)\.", str(name))
    raw = match.group(1).strip() if match else "Rare"
    return TITLE_MAP.get(raw, "Rare")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build predictive features from raw Kaggle Titanic columns.

    Adds title, family structure, deck, and log-fare signals that historically
    improve survival models beyond the raw passenger fields.
    """
    out = df.copy()

    out["Title"] = out["Name"].map(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["Deck"] = (
        out["Cabin"]
        .astype(str)
        .str[0]
        .replace({"n": "Unknown", "T": "Unknown"})
        .fillna("Unknown")
    )
    # Fare is heavily right-skewed; log1p stabilizes scale for linear models.
    out["LogFare"] = np.log1p(out["Fare"].fillna(out["Fare"].median()))

    # Group-aware age fill used when callers skip model-based imputation.
    age_medians = out.groupby(["Sex", "Pclass", "Title"])["Age"].transform("median")
    global_median = out["Age"].median()
    out["AgeFilled"] = out["Age"].fillna(age_medians).fillna(global_median)

    return out


def prepare_model_matrix(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode categoricals and return a numeric design matrix.

    Default feature set balances interpretability with predictive power.
    """
    if feature_cols is None:
        feature_cols = [
            "Pclass",
            "Sex",
            "AgeFilled",
            "SibSp",
            "Parch",
            "LogFare",
            "Embarked",
            "Title",
            "FamilySize",
            "IsAlone",
            "HasCabin",
        ]

    frame = df.loc[:, list(feature_cols)].copy()
    known_cats = {"Sex", "Embarked", "Title", "Deck"}
    categorical = []
    for c in frame.columns:
        series = frame[c]
        is_text_like = (
            c in known_cats
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or str(series.dtype) in {"string", "category", "str"}
        )
        # Avoid treating boolean / numeric columns as categories.
        if is_text_like and not pd.api.types.is_bool_dtype(series):
            if c in known_cats or not pd.api.types.is_numeric_dtype(series):
                categorical.append(c)
    if categorical:
        frame = pd.get_dummies(frame, columns=categorical, drop_first=drop_first)

    return frame.astype(float)
