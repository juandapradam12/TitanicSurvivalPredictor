# Tuning & calibration

## Hyperparameter search

`scripts/tune_model.py` runs a **RandomizedSearchCV** over leak-safe Pipelines
for `sklearn_logistic`, `gradient_boosting`, or `random_forest`.

```bash
python3 scripts/tune_model.py --model gradient_boosting --n-iter 20
```

Artifacts:

- `artifacts/tuned_model.joblib` — best Pipeline refit on all training data
- `artifacts/tuning_results.json` — best params + top CV rows

Search spaces live in `src/titanic_predictor/tuning.py` (`SEARCH_SPACES`).

## Probability calibration

Tree ensembles and even logistic models can be miscalibrated. Use isotonic or
Platt (`sigmoid`) calibration:

```bash
python3 scripts/calibrate_and_threshold.py --model gradient_boosting --method isotonic
```

This script:

1. Holds out 20% of passengers
2. Reports Brier score / ROC-AUC / F1 at threshold 0.5
3. Sweeps thresholds to maximize F1
4. Fits `CalibratedClassifierCV` and repeats
5. Writes reliability + ROC/PR figures under `figures/`
6. Saves a full-data calibrated model to `artifacts/calibrated_model.joblib`

Serve predictions with the recommended threshold from
`artifacts/calibration_results.json` (also exposed as a slider in the Streamlit
demo).

## Streamlit demo

```bash
streamlit run app.py
```

The app loads `calibrated_model.joblib` if present, otherwise `tuned_model` /
`best_model`, otherwise trains a Gradient Boosting Pipeline on the fly.
