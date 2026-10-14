#!/usr/bin/env python3
"""Train on train.csv and write a Kaggle-format submission.csv for test.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.models import build_model_zoo  # noqa: E402
from titanic_predictor.pipeline import (  # noqa: E402
    load_titanic,
    predict_submission,
    run_benchmark,
    save_model,
    train_final_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "test.csv")
    parser.add_argument(
        "--model",
        default="auto",
        choices=[
            "auto",
            "custom_logistic",
            "sklearn_logistic",
            "random_forest",
            "gradient_boosting",
        ],
        help="Model to train. 'auto' picks the best CV ROC-AUC model.",
    )
    parser.add_argument(
        "--age-strategy",
        default="knn",
        choices=["knn", "group_median"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submission.csv",
        help="Path for Kaggle submission CSV",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "artifacts" / "best_model.joblib",
        help="Where to persist the fitted pipeline",
    )
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    if not args.test.exists():
        raise SystemExit(
            f"Missing {args.test}. Place Kaggle's test.csv in the repo root."
        )

    model_name = args.model
    if model_name == "auto":
        print("Selecting model via stratified CV…")
        bench = run_benchmark(
            args.train,
            cv_splits=args.cv,
            age_strategy=args.age_strategy,
        )
        model_name = bench["best_model"]
        print(bench["leaderboard"].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        print(f"\nSelected: {model_name}")

    print(f"Fitting {model_name} on full training set…")
    model, _, _ = train_final_model(
        args.train,
        model_name=model_name,
        age_strategy=args.age_strategy,
    )
    submission = predict_submission(model, args.test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    save_model(model, args.model_out)

    print(f"Wrote submission → {args.out}  ({len(submission)} rows)")
    print(f"Wrote model      → {args.model_out}")
    print(submission.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
