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

## What changed through the 1.3 refresh

| Area | Before | After |
|---|---|---|
| Evaluation | In-sample classification report | Stratified CV + holdout |
| Custom LR | Slow Python loops, fixed steps | Vectorized NumPy, early stopping, sklearn API |
| Features | Raw dummies | Title / family / cabin / log-fare |
| Imputation | Global (pre-CV) k-NN | **Fold-safe** k-NN / group-median inside Pipelines |
| Submission | None | `make_submission.py` → `submission.csv` + joblib artifact |
| Interpretation | None | Coefficients + SHAP CLI / API |
| Tuning | Fixed defaults | RandomizedSearchCV spaces + CLI |
| Calibration | Threshold 0.5 only | Isotonic/Platt + F1-optimal threshold |
| Demo / CI | None | Streamlit app + GitHub Actions |

## How to extend

- Swap in `HistGradientBoostingClassifier` or Optuna for heavier search.
- Deploy the Streamlit app (Streamlit Community Cloud / Hugging Face Spaces).
- Add fare-per-person / ticket-group features if chasing leaderboard points.
