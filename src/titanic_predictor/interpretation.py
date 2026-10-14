"""Coefficient tables and SHAP-based explanations for fitted pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _final_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps[list(model.named_steps)[-1]]
    return model


def _transformed_matrix(model, X) -> Tuple[pd.DataFrame, Any]:
    """
    Apply all pipeline steps except the final estimator.

    Returns a dense feature matrix aligned with the classifier coefficients /
    SHAP inputs, plus the final estimator.
    """
    if not hasattr(model, "named_steps"):
        raise ValueError("Expected a scikit-learn Pipeline.")

    steps = list(model.named_steps.items())
    Xt = X
    for name, step in steps[:-1]:
        Xt = step.transform(Xt)
    if not isinstance(Xt, pd.DataFrame):
        Xt = pd.DataFrame(Xt)
    return Xt, steps[-1][1]


def logistic_coefficient_table(model, feature_names: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Extract intercept + coefficients for logistic-style final estimators.

    Works with ``CustomLogisticRegression`` and sklearn ``LogisticRegression``
    (including when nested inside a Pipeline).
    """
    est = _final_estimator(model)
    if not hasattr(est, "coef_"):
        raise ValueError("Final estimator has no coef_ attribute.")

    coefs = np.asarray(est.coef_).reshape(-1)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(coefs))]
    intercept = float(np.asarray(est.intercept_).reshape(-1)[0]) if hasattr(est, "intercept_") else 0.0

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)
    table.attrs["intercept"] = intercept
    return table.reset_index(drop=True)


def explain_with_shap(
    model,
    X,
    *,
    max_samples: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute SHAP values for a fitted Pipeline.

    Uses ``LinearExplainer`` for logistic models and ``TreeExplainer`` for
    tree ensembles. Falls back to ``PermutationExplainer`` when needed.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shap is required for explain_with_shap. Install with: pip install shap"
        ) from exc

    Xt, estimator = _transformed_matrix(model, X)
    if len(Xt) > max_samples:
        Xt_sample = Xt.sample(n=max_samples, random_state=random_state)
    else:
        Xt_sample = Xt

    name = estimator.__class__.__name__.lower()
    if "logistic" in name:
        explainer = shap.LinearExplainer(estimator, Xt_sample)
        shap_values = explainer(Xt_sample)
    elif any(k in name for k in ("forest", "boost", "tree")):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(Xt_sample)
    else:
        explainer = shap.Explainer(estimator.predict_proba, Xt_sample)
        shap_values = explainer(Xt_sample)

    # Normalize to a 2D array of mean |SHAP| per feature for the positive class.
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs = np.mean(np.abs(values), axis=0)
    importance = (
        pd.DataFrame(
            {"feature": list(Xt_sample.columns), "mean_abs_shap": mean_abs}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "explainer_type": type(explainer).__name__,
        "shap_values": shap_values,
        "X_sample": Xt_sample,
        "importance": importance,
    }


def plot_coefficient_bars(
    coef_table: pd.DataFrame,
    *,
    top_n: int = 15,
    output_path: Optional[Path | str] = None,
    title: str = "Logistic coefficients (largest magnitude)",
):
    """Horizontal bar chart of the top-|coef| features."""
    import matplotlib.pyplot as plt

    plot_df = coef_table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(plot_df) + 1.5))
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in plot_df["coefficient"]]
    ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
    return fig


def plot_shap_importance(
    importance: pd.DataFrame,
    *,
    top_n: int = 15,
    output_path: Optional[Path | str] = None,
    title: str = "Mean |SHAP| feature importance",
):
    """Bar chart of mean absolute SHAP values."""
    import matplotlib.pyplot as plt

    plot_df = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(plot_df) + 1.5))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2c6eaa")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
    return fig


def narrative_from_coefficients(coef_table: pd.DataFrame, top_n: int = 5) -> str:
    """Short plain-language summary of the strongest linear drivers."""
    top = coef_table.head(top_n)
    lines = [
        "Strongest linear drivers of predicted survival (by |coefficient|):",
        "",
    ]
    for _, row in top.iterrows():
        direction = "increases" if row["coefficient"] > 0 else "decreases"
        lines.append(
            f"- **{row['feature']}** {direction} survival odds "
            f"(coef={row['coefficient']:.3f}, OR={row['odds_ratio']:.3f})."
        )
    return "\n".join(lines)
