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

## What changed in the 1.1 refresh

| Area | Before | After |
|---|---|---|
| Evaluation | In-sample classification report | Stratified CV + holdout |
| Custom LR | Slow Python loops, fixed steps | Vectorized NumPy, early stopping, sklearn API |
| Features | Raw dummies | Title / family / cabin / log-fare |
| Imputation | Unscaled k-NN | Scaled k-NN (+ group-median option) |
| Repo surface | Notebook + CSV + stub README | Package, docs, tests, CLI, selling README |

## How to extend

- Add Kaggle `test.csv` scoring in `scripts/` using `build_xy`’s feature path.
- Swap in `HistGradientBoostingClassifier` or Optuna tuning for the tree models.
- Export coefficient tables / SHAP summaries for stakeholder-facing reports.
