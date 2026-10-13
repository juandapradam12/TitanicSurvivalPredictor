#!/usr/bin/env python3
"""Run the Titanic Survival Predictor benchmark and print a leaderboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.pipeline import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "train.csv",
        help="Path to Titanic train.csv",
    )
    parser.add_argument("--cv", type=int, default=5, help="Stratified CV folds")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable summary JSON",
    )
    args = parser.parse_args()

    result = run_benchmark(args.data, cv_splits=args.cv)
    board = result["leaderboard"]

    print("=" * 64)
    print(" Titanic Survival Predictor — Model Benchmark")
    print("=" * 64)
    print(f"Samples: {result['n_samples']}  |  Features: {result['n_features']}")
    print(f"Best model (by ROC-AUC CV): {result['best_model']}")
    print()
    print(board.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print()
    print("Holdout report for best model")
    print("-" * 32)
    holdout = result["holdout"]
    print(f"accuracy : {holdout['accuracy']:.4f}")
    print(f"f1       : {holdout['f1']:.4f}")
    if "roc_auc" in holdout:
        print(f"roc_auc  : {holdout['roc_auc']:.4f}")
    print()
    print(holdout["classification_report"])

    if args.json_out:
        payload = {
            "n_samples": result["n_samples"],
            "n_features": result["n_features"],
            "feature_names": result["feature_names"],
            "best_model": result["best_model"],
            "leaderboard": board.to_dict(orient="records"),
            "holdout": {
                k: v
                for k, v in holdout.items()
                if k != "classification_report"
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote summary → {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
