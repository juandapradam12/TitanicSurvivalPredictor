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
from titanic_predictor.models import build_model_zoo
from titanic_predictor.pipeline import (
    build_xy,
    interpret_model,
    load_titanic,
    predict_submission,
    run_benchmark,
    train_final_model,
)
from titanic_predictor.preprocessing import (
    build_preprocess_pipeline,
    prepare_raw_xy,
)


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


def test_build_xy_returns_raw_columns():
    df = load_titanic(ROOT / "train.csv")
    X, y = build_xy(df, use_knn_age=True)
    assert len(X) == len(y)
    assert "Age" in X.columns and "Name" in X.columns
    assert set(y.unique()) <= {0, 1}


def test_preprocess_pipeline_no_nans():
    df = load_titanic(ROOT / "train.csv")
    X, y = prepare_raw_xy(df)
    pipe = build_preprocess_pipeline(age_strategy="knn", scale_numeric=True)
    Xt = pipe.fit_transform(X, y)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()
    assert Xt.shape[0] == len(X)


def test_leak_safe_pipeline_fit_transform_shapes():
    """Imputer fitted on train must transform a disjoint test fold."""
    df = load_titanic(ROOT / "train.csv")
    X, y = prepare_raw_xy(df)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train = y.iloc[:split]
    model = build_model_zoo(age_strategy="group_median")["sklearn_logistic"]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_benchmark_smoke():
    result = run_benchmark(
        ROOT / "train.csv",
        cv_splits=3,
        test_size=0.25,
        age_strategy="group_median",
    )
    assert result["n_samples"] > 800
    assert result["best_model"] in {
        "custom_logistic",
        "sklearn_logistic",
        "random_forest",
        "gradient_boosting",
    }
    assert result["holdout"]["accuracy"] > 0.7
    assert not result["leaderboard"].empty


def test_submission_shape():
    model, _, _ = train_final_model(
        ROOT / "train.csv",
        model_name="sklearn_logistic",
        age_strategy="group_median",
    )
    submission = predict_submission(model, ROOT / "test.csv")
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert len(submission) == len(pd.read_csv(ROOT / "test.csv"))
    assert set(submission["Survived"].unique()).issubset({0, 1})


def test_interpret_logistic():
    model, X, _ = train_final_model(
        ROOT / "train.csv",
        model_name="sklearn_logistic",
        age_strategy="group_median",
    )
    # Use a small sample for speed in CI-like runs.
    bundle = interpret_model(model, X.iloc[:120], max_shap_samples=80)
    assert "coefficients" in bundle
    assert "shap_importance" in bundle
    assert bundle["shap_importance"]["mean_abs_shap"].iloc[0] >= 0


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


def test_tune_logistic_smoke():
    from titanic_predictor.tuning import tune_model

    df = load_titanic(ROOT / "train.csv")
    X, y = prepare_raw_xy(df)
    result = tune_model(
        X,
        y,
        model_name="sklearn_logistic",
        age_strategy="group_median",
        n_iter=4,
        cv_splits=3,
    )
    assert result["best_score"] > 0.8
    assert "model__C" in result["best_params"]
    preds = result["best_estimator"].predict(X.iloc[:20])
    assert len(preds) == 20


def test_calibration_and_threshold_smoke():
    from titanic_predictor.calibration import calibration_report, find_best_threshold

    df = load_titanic(ROOT / "train.csv")
    X, y = prepare_raw_xy(df)
    model = build_model_zoo(age_strategy="group_median")["sklearn_logistic"]
    report = calibration_report(model, X, y, method="sigmoid")
    assert report["calibrated_at_0_5"]["brier"] < 0.25
    assert 0.05 <= report["calibrated_best_threshold"]["best_threshold"] <= 0.95

    thresh = find_best_threshold(
        report["y_test"], report["calibrated_proba"], metric="f1"
    )
    assert thresh["best_score"] >= report["calibrated_at_0_5"]["f1"] - 1e-9
