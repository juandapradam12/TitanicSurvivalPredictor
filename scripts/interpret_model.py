#!/usr/bin/env python3
"""Fit a model and export coefficient / SHAP interpretation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.interpretation import (  # noqa: E402
    plot_coefficient_bars,
    plot_shap_importance,
)
from titanic_predictor.pipeline import (  # noqa: E402
    interpret_model,
    train_final_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "train.csv")
    parser.add_argument(
        "--model",
        default="sklearn_logistic",
        choices=[
            "custom_logistic",
            "sklearn_logistic",
            "random_forest",
            "gradient_boosting",
        ],
    )
    parser.add_argument(
        "--age-strategy",
        default="knn",
        choices=["knn", "group_median"],
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures",
    )
    parser.add_argument(
        "--max-shap-samples",
        type=int,
        default=200,
    )
    args = parser.parse_args()

    model, X, _y = train_final_model(
        args.train,
        model_name=args.model,
        age_strategy=args.age_strategy,
    )
    bundle = interpret_model(
        model,
        X,
        max_shap_samples=args.max_shap_samples,
    )

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if "coefficients" in bundle:
        coef_path = args.figures_dir / f"coefficients_{args.model}.png"
        plot_coefficient_bars(
            bundle["coefficients"],
            output_path=coef_path,
            title=f"{args.model} — logistic coefficients",
        )
        print(bundle["coefficient_narrative"])
        print(f"\nWrote {coef_path}")
        print(bundle["coefficients"].head(10).to_string(index=False))
        print()

    shap_path = args.figures_dir / f"shap_importance_{args.model}.png"
    plot_shap_importance(
        bundle["shap_importance"],
        output_path=shap_path,
        title=f"{args.model} — mean |SHAP|",
    )
    print(f"SHAP explainer: {bundle['shap_explainer_type']}")
    print(bundle["shap_importance"].head(10).to_string(index=False))
    print(f"\nWrote {shap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
