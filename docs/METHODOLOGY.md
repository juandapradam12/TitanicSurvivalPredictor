# Methodology

## Problem

Predict passenger survival on the RMS Titanic from demographic and voyage
attributes. The label is binary (`Survived ∈ {0, 1}`), so the natural framing
is **supervised binary classification** with an emphasis on **interpretability**.

Data source: [Kaggle Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data).

## Pipeline overview (leak-safe)

```text
raw CSV rows
  → TitanicFeatureBuilder   (fit: Embarked mode, Fare median)
  → Age imputer             (fit: k-NN or Sex×Pclass×Title medians)
  → OneHotEncoder + scale   (fit: categories / numeric stats)
  → classifier
```

Every step above lives inside a scikit-learn `Pipeline`. Stratified k-fold CV
therefore **refits imputation and encoding on each training fold only** — the
holdout fold never influences those statistics.

## Feature engineering

| Feature | Rationale |
|---|---|
| `Title` | Honorifics encode social status, gender, and age band (e.g. Master vs Mr). |
| `FamilySize` / `IsAlone` | Traveling alone vs in a large family correlates with survival odds. |
| `HasCabin` | Cabin assignment proxies for class / deck access; mostly missing otherwise. |
| `LogFare` | Fare is right-skewed; log transform improves linear-model conditioning. |
| `Age` | Imputed inside the Pipeline (k-NN or group median). |

## Age imputation

Two strategies, both fold-safe:

1. **Scaled k-NN regression** (`age_strategy="knn"`): neighbors vote on Age
   using voyage / demographic predictors standardized on the training fold.
2. **Group medians** (`age_strategy="group_median"`): Sex×Pclass×Title medians
   learned on the training fold, with a global median fallback.

## Evaluation protocol

- **Stratified 5-fold cross-validation** for leaderboard metrics
  (`accuracy`, `F1`, `ROC-AUC`).
- A stratified **20% holdout** report for the best CV model.
- Stratification preserves the ~38% survival base rate in every fold.

## Model families

| Model | Role |
|---|---|
| Custom logistic regression | Pedagogical from-scratch MLE / gradient ascent. |
| scikit-learn logistic regression | Strong, regularized linear baseline. |
| Random Forest | Non-linear interactions without heavy tuning. |
| Gradient Boosting | Typically strongest tabular baseline on this dataset. |

## Kaggle submission

```bash
python3 scripts/make_submission.py --model auto
```

Selects the best CV model (or a named one), fits on the full `train.csv`,
scores `test.csv`, and writes `submission.csv` plus a persisted
`artifacts/best_model.joblib`.

## Interpretation

See [INTERPRETATION.md](INTERPRETATION.md) for coefficient tables and SHAP.

See [ALGORITHMS.md](ALGORITHMS.md) for the math behind the custom estimator.
