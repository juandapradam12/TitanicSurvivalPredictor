#!/usr/bin/env python3
"""Calibrate probabilities, tune decision threshold, and export diagnostic plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.calibration import (  # noqa: E402
    calibration_report,
    plot_reliability,
    plot_roc_pr,
)
from titanic_predictor.models import build_model_zoo  # noqa: E402
from titanic_predictor.pipeline import load_titanic, save_model  # noqa: E402
from titanic_predictor.preprocessing import prepare_raw_xy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "train.csv")
    parser.add_argument(
        "--model",
        default="gradient_boosting",
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
        "--method",
        default="isotonic",
        choices=["isotonic", "sigmoid"],
    )
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "artifacts" / "calibrated_model.joblib",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "artifacts" / "calibration_results.json",
    )
    args = parser.parse_args()

    df = load_titanic(args.train)
    X, y = prepare_raw_xy(df)
    assert y is not None

    base = build_model_zoo(age_strategy=args.age_strategy)[args.model]
    report = calibration_report(base, X, y, method=args.method)

    print("=" * 60)
    print(f" Calibration & threshold report — {args.model}")
    print("=" * 60)
    print("\nUncalibrated @ 0.5:")
    for k, v in report["raw_at_0_5"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(
        f"\nUncalibrated best F1 threshold: "
        f"{report['raw_best_threshold']['best_threshold']:.2f} "
        f"(F1={report['raw_best_threshold']['best_score']:.4f})"
    )
    print("Uncalibrated @ best threshold:")
    for k, v in report["raw_at_best_threshold"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nCalibrated @ 0.5:")
    for k, v in report["calibrated_at_0_5"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(
        f"\nCalibrated best F1 threshold: "
        f"{report['calibrated_best_threshold']['best_threshold']:.2f} "
        f"(F1={report['calibrated_best_threshold']['best_score']:.4f})"
    )
    print("Calibrated @ best threshold:")
    for k, v in report["calibrated_at_best_threshold"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    rel_path = args.figures_dir / f"reliability_{args.model}.png"
    roc_path = args.figures_dir / f"roc_pr_{args.model}.png"
    plot_reliability(
        report["y_test"],
        report["raw_proba"],
        report["calibrated_proba"],
        output_path=rel_path,
        title=f"{args.model} — reliability",
    )
    plot_roc_pr(
        report["y_test"],
        report["calibrated_proba"],
        output_path=roc_path,
        title_prefix=f"{args.model} calibrated",
    )

    # Refit calibrated model on full data for serving.
    from sklearn.base import clone
    from titanic_predictor.calibration import calibrate_model

    full_calibrated = calibrate_model(
        clone(base), X, y, method=args.method
    )
    save_model(full_calibrated, args.model_out)

    meta = {
        "model": args.model,
        "calibration_method": args.method,
        "raw_at_0_5": report["raw_at_0_5"],
        "raw_best_threshold": {
            "threshold": report["raw_best_threshold"]["best_threshold"],
            "f1": report["raw_best_threshold"]["best_score"],
        },
        "raw_at_best_threshold": report["raw_at_best_threshold"],
        "calibrated_at_0_5": report["calibrated_at_0_5"],
        "calibrated_best_threshold": {
            "threshold": report["calibrated_best_threshold"]["best_threshold"],
            "f1": report["calibrated_best_threshold"]["best_score"],
        },
        "calibrated_at_best_threshold": report["calibrated_at_best_threshold"],
        "recommended_threshold": report["calibrated_best_threshold"]["best_threshold"],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {rel_path}")
    print(f"Wrote {roc_path}")
    print(f"Wrote model → {args.model_out}")
    print(f"Wrote meta  → {args.json_out}")
    print(
        f"\nRecommended serving threshold: {meta['recommended_threshold']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
