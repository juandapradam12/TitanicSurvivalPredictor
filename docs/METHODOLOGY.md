# Methodology

## Problem

Predict passenger survival on the RMS Titanic from demographic and voyage
attributes. The label is binary (`Survived ∈ {0, 1}`), so the natural framing
is **supervised binary classification** with an emphasis on **interpretability**.

Data source: [Kaggle Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data).

## Pipeline overview

```text
raw CSV
  → drop rows with missing Embarked (2 passengers)
  → feature engineering (title, family size, cabin presence, log-fare, …)
  → Age imputation (scaled k-NN or group medians)
  → one-hot encoding of categoricals
  → stratified train/test + k-fold cross-validation
  → model zoo comparison
```

## Feature engineering

| Feature | Rationale |
|---|---|
| `Title` | Honorifics encode social status, gender, and age band (e.g. Master vs Mr). |
| `FamilySize` / `IsAlone` | Traveling alone vs in a large family correlates with survival odds. |
| `HasCabin` | Cabin assignment proxies for class / deck access; mostly missing otherwise. |
| `LogFare` | Fare is right-skewed; log transform improves linear-model conditioning. |
| `AgeFilled` | Age is informative but ~20% missing; imputed before modeling. |

## Age imputation

Two complementary strategies are supported:

1. **Scaled k-NN regression** (classic notebook path): neighbors vote on Age using
   the remaining numeric predictors after `StandardScaler`.
2. **Group medians** by `(Sex, Pclass, Title)`: fast, robust baseline used when
   a parametric imputer is unnecessary.

## Evaluation protocol

The original notebook reported metrics on the **same data used for training**.
That overstates performance. This revision evaluates models with:

- **Stratified 5-fold cross-validation** for leaderboard metrics (`accuracy`,
  `F1`, `ROC-AUC`).
- A stratified **20% holdout** report for the best CV model.

Stratification preserves the ~38% survival base rate in every fold.

## Model families

| Model | Role |
|---|---|
| Custom logistic regression | Pedagogical from-scratch MLE / gradient ascent. |
| scikit-learn logistic regression | Strong, regularized linear baseline. |
| Random Forest | Non-linear interactions without heavy tuning. |
| Gradient Boosting | Typically strongest tabular baseline on this dataset. |

See [ALGORITHMS.md](ALGORITHMS.md) for the math behind the custom estimator.
