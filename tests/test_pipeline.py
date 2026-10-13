"""Smoke tests for the Titanic Survival Predictor package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.custom_logistic import CustomLogisticRegression
from titanic_predictor.features import engineer_features, extract_title
from titanic_predictor.pipeline import build_xy, load_titanic, run_benchmark


def test_extract_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Cumings, Mrs. John Bradley (Florence Briggs Thayer)") == "Mrs"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Fortune, Master. Richard") == "Master"


def test_custom_logistic_learns_separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = CustomLogisticRegression(learning_rate=0.8, max_iter=2000)
    model.fit(X, y)
    assert model.score(X, y) > 0.85


def test_build_xy_shapes():
    df = load_titanic(ROOT / "train.csv")
    X, y = build_xy(df, use_knn_age=True)
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}


def test_benchmark_smoke():
    result = run_benchmark(ROOT / "train.csv", cv_splits=3, test_size=0.25)
    assert result["n_samples"] > 800
    assert result["best_model"] in {
        "custom_logistic",
        "sklearn_logistic",
        "random_forest",
        "gradient_boosting",
    }
    assert result["holdout"]["accuracy"] > 0.7
    assert not result["leaderboard"].empty


def test_engineer_features_adds_columns():
    df = pd.DataFrame(
        {
            "Name": ["Smith, Mr. John", "Doe, Miss. Jane"],
            "Sex": ["male", "female"],
            "Pclass": [3, 1],
            "Age": [30.0, np.nan],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Fare": [7.25, 71.0],
            "Cabin": [np.nan, "C85"],
            "Embarked": ["S", "C"],
        }
    )
    out = engineer_features(df)
    for col in ["Title", "FamilySize", "IsAlone", "HasCabin", "LogFare", "AgeFilled"]:
        assert col in out.columns
