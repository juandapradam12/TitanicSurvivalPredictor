# Project overview

## Elevator pitch

A Titanic survival classifier that teaches **maximum-likelihood logistic
regression from first principles**, then stress-tests that estimator against
modern scikit-learn baselines under an honest cross-validation protocol.

## Audience

- Hiring managers / reviewers scanning for ML fundamentals
- Students learning the link between likelihood, gradient ascent, and `predict_proba`
- Practitioners who want a clean reference pipeline for tabular binary classification

## Design principles

1. **Transparency over magic** — the custom logistic model exposes coefficients and training history.
2. **Fair comparison** — every model sees the same engineered features and stratified folds.
3. **Portfolio-ready packaging** — notebook for narrative, `src/` for reuse, CLI for one-command demos, tests for regression safety.

## What changed in the 1.1 → 1.2 refresh

| Area | Before | After |
|---|---|---|
| Evaluation | In-sample classification report | Stratified CV + holdout |
| Custom LR | Slow Python loops, fixed steps | Vectorized NumPy, early stopping, sklearn API |
| Features | Raw dummies | Title / family / cabin / log-fare |
| Imputation | Global (pre-CV) k-NN | **Fold-safe** k-NN / group-median inside Pipelines |
| Submission | None | `make_submission.py` → `submission.csv` + joblib artifact |
| Interpretation | None | Coefficients + SHAP CLI / API |
| Repo surface | Notebook + CSV + stub README | Package, docs, tests, CLIs, selling README |

## How to extend

- Swap in `HistGradientBoostingClassifier` or Optuna tuning for the tree models.
- Add probability calibration and threshold optimization for F1 / business costs.
- Publish a Streamlit demo that loads `artifacts/best_model.joblib`.
