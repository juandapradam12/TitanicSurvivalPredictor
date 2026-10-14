"""
Leak-safe preprocessing transformers for Titanic modeling.

All statistics (embarked mode, fare median, age imputation, one-hot categories,
scaling) are learned in ``fit`` on the training fold only and applied in
``transform``. Use these inside a scikit-learn ``Pipeline`` so cross-validation
does not peek at holdout rows.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import extract_title


RAW_COLUMNS = [
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Cabin",
    "Embarked",
]

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "LogFare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
]

CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]


class TitanicFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Derive modeling columns from raw Kaggle fields.

    Learns Embarked mode and Fare median on the training fold; leaves Age
    missing so a downstream imputer can fit without leakage.
    """

    def fit(self, X, y=None):
        frame = self._as_frame(X)
        self.embarked_mode_ = (
            frame["Embarked"].dropna().mode().iloc[0]
            if frame["Embarked"].notna().any()
            else "S"
        )
        self.fare_median_ = float(frame["Fare"].median())
        return self

    def transform(self, X):
        frame = self._as_frame(X).copy()
        frame["Embarked"] = frame["Embarked"].fillna(self.embarked_mode_)
        frame["Fare"] = frame["Fare"].fillna(self.fare_median_)
        frame["Title"] = frame["Name"].map(extract_title)
        frame["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1
        frame["IsAlone"] = (frame["FamilySize"] == 1).astype(int)
        frame["HasCabin"] = frame["Cabin"].notna().astype(int)
        frame["LogFare"] = np.log1p(frame["Fare"].astype(float))
        # Keep Age as-is (may contain NaNs) for the imputer stage.
        keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        return frame.loc[:, keep]

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=RAW_COLUMNS)


class GroupMedianAgeImputer(BaseEstimator, TransformerMixin):
    """Impute Age using training medians by (Sex, Pclass, Title)."""

    def fit(self, X, y=None):
        frame = self._as_frame(X)
        grouped = (
            frame.groupby(["Sex", "Pclass", "Title"])["Age"]
            .median()
            .dropna()
        )
        self.group_medians_ = grouped.to_dict()
        self.global_median_ = float(frame["Age"].median())
        if np.isnan(self.global_median_):
            self.global_median_ = 28.0
        return self

    def transform(self, X):
        frame = self._as_frame(X).copy()

        def _fill(row):
            if pd.notna(row["Age"]):
                return row["Age"]
            key = (row["Sex"], row["Pclass"], row["Title"])
            return self.group_medians_.get(key, self.global_median_)

        frame["Age"] = frame.apply(_fill, axis=1)
        return frame

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        return pd.DataFrame(X, columns=cols)


class KNNAgeImputer(BaseEstimator, TransformerMixin):
    """
    Scaled k-NN Age imputer fitted only on training rows with observed Age.

    Uses Sex/Embarked/Title one-hots plus numeric voyage fields as neighbors.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        frame = GroupMedianAgeImputer._as_frame(X)
        known = frame[frame["Age"].notna()]
        if known.empty:
            raise ValueError("Cannot fit KNNAgeImputer without any observed ages.")

        design, self.dummy_columns_ = self._design_matrix(known, fit=True)
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(design)
        self.model_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.model_.fit(x_scaled, known["Age"].astype(float))
        # Fallback for transform rows if design is empty edge-case.
        self.global_median_ = float(known["Age"].median())
        return self

    def transform(self, X):
        frame = GroupMedianAgeImputer._as_frame(X).copy()
        missing_mask = frame["Age"].isna()
        if not missing_mask.any():
            return frame

        design, _ = self._design_matrix(frame.loc[missing_mask], fit=False)
        preds = self.model_.predict(self.scaler_.transform(design))
        frame.loc[missing_mask, "Age"] = np.clip(preds, 0.0, 80.0)
        frame["Age"] = frame["Age"].fillna(self.global_median_)
        return frame

    def _design_matrix(self, frame: pd.DataFrame, *, fit: bool):
        base = frame.drop(columns=["Age"]).copy()
        dummies = pd.get_dummies(
            base,
            columns=["Sex", "Embarked", "Title"],
            drop_first=True,
        ).astype(float)
        if fit:
            columns = list(dummies.columns)
        else:
            for col in self.dummy_columns_:
                if col not in dummies.columns:
                    dummies[col] = 0.0
            dummies = dummies.loc[:, self.dummy_columns_]
            columns = self.dummy_columns_
        return dummies, columns


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    """ColumnTransformer wrapper that returns a dense float DataFrame."""

    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric = [c for c in NUMERIC_FEATURES if c in frame.columns]
        categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]

        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        self.transformer_ = ColumnTransformer(
            transformers=[
                ("num", Pipeline(numeric_steps), numeric),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="if_binary",
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        self.transformer_.fit(frame, y)
        self.feature_names_ = self._output_names(numeric)
        return self

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        arr = self.transformer_.transform(frame)
        return pd.DataFrame(arr, columns=self.feature_names_, index=frame.index)

    def _output_names(self, numeric: List[str]) -> List[str]:
        cat_names = list(
            self.transformer_.named_transformers_["cat"].get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )
        return list(numeric) + cat_names


def build_preprocess_pipeline(
    *,
    age_strategy: str = "knn",
    scale_numeric: bool = True,
    n_neighbors: int = 5,
) -> Pipeline:
    """
    Feature builder → age imputer → encode/scale.

    Parameters
    ----------
    age_strategy:
        ``"knn"`` for scaled k-NN age imputation, ``"group_median"`` for
        Sex×Pclass×Title medians.
    scale_numeric:
        Whether to standardize numeric columns (recommended for linear models;
        trees are fine either way).
    """
    if age_strategy == "knn":
        age_imputer: BaseEstimator = KNNAgeImputer(n_neighbors=n_neighbors)
    elif age_strategy == "group_median":
        age_imputer = GroupMedianAgeImputer()
    else:
        raise ValueError(f"Unknown age_strategy: {age_strategy}")

    return Pipeline(
        steps=[
            ("features", TitanicFeatureBuilder()),
            ("age", age_imputer),
            ("encode", DataFrameColumnTransformer(scale_numeric=scale_numeric)),
        ]
    )


def prepare_raw_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extract raw modeling columns and optional Survived target.

    Drops rows with missing Embarked only when a target is present and those
    rows cannot be recovered; for inference, Embarked is filled in the pipeline.
    """
    frame = df.copy()
    y = None
    if "Survived" in frame.columns:
        # Keep all labeled rows; Embarked nulls are filled inside the pipeline.
        y = frame["Survived"].astype(int)
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = frame.loc[:, RAW_COLUMNS].copy()
    return X, y
