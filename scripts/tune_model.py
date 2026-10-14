#!/usr/bin/env python3
"""Randomized hyperparameter search for a named Titanic model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.pipeline import load_titanic, save_model  # noqa: E402
from titanic_predictor.preprocessing import prepare_raw_xy  # noqa: E402
from titanic_predictor.tuning import SEARCH_SPACES, tune_model  # noqa: E402


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "train.csv")
    parser.add_argument(
        "--model",
        default="gradient_boosting",
        choices=sorted(SEARCH_SPACES),
    )
    parser.add_argument(
        "--age-strategy",
        default="knn",
        choices=["knn", "group_median"],
    )
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "artifacts" / "tuned_model.joblib",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "artifacts" / "tuning_results.json",
    )
    args = parser.parse_args()

    df = load_titanic(args.train)
    X, y = prepare_raw_xy(df)
    assert y is not None

    print(f"Tuning {args.model}  (n_iter={args.n_iter}, cv={args.cv})")
    result = tune_model(
        X,
        y,
        model_name=args.model,
        age_strategy=args.age_strategy,
        n_iter=args.n_iter,
        cv_splits=args.cv,
    )

    print(f"Best CV {result['scoring']}: {result['best_score']:.4f}")
    print("Best params:")
    for k, v in sorted(result["best_params"].items()):
        print(f"  {k}: {v}")
    print()
    print(result["cv_results"].head(5).to_string(index=False))

    save_model(result["best_estimator"], args.model_out)
    top = result["cv_results"].head(10).copy()
    top["params"] = top["params"].map(_jsonable)
    payload = {
        "model_name": result["model_name"],
        "scoring": result["scoring"],
        "best_score": result["best_score"],
        "best_params": _jsonable(result["best_params"]),
        "top_results": top.to_dict(orient="records"),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote model  → {args.model_out}")
    print(f"Wrote summary → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
