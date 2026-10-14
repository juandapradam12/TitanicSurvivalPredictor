"""Titanic Survival Predictor — interpretable binary classification toolkit."""

from .custom_logistic import CustomLogisticRegression
from .features import engineer_features, prepare_model_matrix
from .pipeline import interpret_model, predict_submission, run_benchmark

__all__ = [
    "CustomLogisticRegression",
    "engineer_features",
    "prepare_model_matrix",
    "run_benchmark",
    "predict_submission",
    "interpret_model",
]

__version__ = "1.3.0"
