# Titanic Survival Predictor

**Interpretable binary classification with a from-scratch logistic regression core — plus a modern model benchmark on Kaggle’s Titanic dataset.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/data-Kaggle%20Titanic-lightgrey.svg)](https://www.kaggle.com/c/titanic/data)

---

### Why this project exists

Most Titanic tutorials call `LogisticRegression().fit(...)` and stop. This repository goes further:

1. **Derive and implement** logistic regression from the likelihood and its gradient.
2. **Impute missing ages** with k-nearest neighbors (scaled), not a blunt global mean.
3. **Engineer features** that encode social structure (title, family size, cabin presence).
4. **Compare models fairly** with stratified cross-validation and a holdout set — not in-sample scores.

The result is a portfolio piece that shows *statistical understanding*, *clean engineering*, and *honest evaluation*.

---

## Highlights

| Capability | Detail |
|---|---|
| Custom MLE logistic regression | Vectorized gradient ascent, scikit-learn-compatible API |
| Feature engineering | Title, FamilySize, IsAlone, HasCabin, LogFare |
| Age imputation | Scaled k-NN regressor (+ group-median fallback) |
| Model zoo | Custom LR · sklearn LR · Random Forest · Gradient Boosting |
| Evaluation | Stratified 5-fold CV (Accuracy / F1 / ROC-AUC) + holdout report |
| Reproducible CLI | `python scripts/train_and_evaluate.py` |
| Walkthrough notebook | Math, EDA, and benchmark in `TitanicSurvivalPredictor.ipynb` |

---

## Quick start

```bash
# Clone
git clone https://github.com/juandapradam12/TitanicSurvivalPredictor.git
cd TitanicSurvivalPredictor

# Install
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the benchmark
python3 scripts/train_and_evaluate.py

# Or open the notebook
jupyter notebook TitanicSurvivalPredictor.ipynb
```

On the engineered feature set, stratified 5-fold CV typically lands around **~83–84% accuracy** and **~0.87 ROC-AUC**. Gradient Boosting usually leads the leaderboard; the **custom logistic regression stays within ~1 point of scikit-learn’s**, which is the point of the exercise — interpretability without giving up much signal.

---

## Project layout

```text
TitanicSurvivalPredictor/
├── TitanicSurvivalPredictor.ipynb   # Narrative walkthrough
├── train.csv                        # Kaggle training data
├── requirements.txt
├── src/titanic_predictor/           # Reusable package
│   ├── custom_logistic.py           # From-scratch logistic regression
│   ├── features.py                  # Feature engineering
│   ├── imputation.py                # k-NN age imputation
│   ├── models.py                    # Model zoo + CV metrics
│   └── pipeline.py                  # End-to-end benchmark
├── scripts/train_and_evaluate.py    # CLI entry point
├── tests/test_pipeline.py           # Smoke tests
└── docs/
    ├── METHODOLOGY.md               # Data, features, evaluation
    └── ALGORITHMS.md                # Likelihood, gradient, upgrades
```

---

## The math (short version)

For labels \(y_i \in \{-1,+1\}\),

$$
\nabla_w \mathcal{L}
  = \sum_{i=1}^{n}
    \big(1 - \sigma(y_i\, x_i^\top w)\big)\, y_i\, x_i,
\quad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Weights are updated by gradient ascent. The package version is **vectorized**, supports **early stopping**, and exposes `coef_`, `predict_proba`, and `predict` like any scikit-learn classifier.

Full derivation: [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

---

## Example output

<p align="center">
  <img src="figures/leaderboard.png" alt="Model leaderboard ROC-AUC" width="720"/>
</p>

```text
================================================================
 Titanic Survival Predictor — Model Benchmark
================================================================
Samples: 889  |  Features: 15
Best model (by ROC-AUC CV): gradient_boosting

            model  accuracy_mean  f1_mean  roc_auc_mean
gradient_boosting         0.8358   0.7729        0.8764
 sklearn_logistic         0.8279   0.7709        0.8749
  custom_logistic         0.8245   0.7653        0.8747
    random_forest         0.8335   0.7695        0.8687
```

*(Figures from a local stratified 5-fold run; re-run the script for your environment.)*

---

## Documentation

- [Methodology](docs/METHODOLOGY.md) — data cleaning, features, CV protocol
- [Algorithms](docs/ALGORITHMS.md) — custom logistic regression & imputation
- [Notebook](TitanicSurvivalPredictor.ipynb) — exploratory analysis and live training

---

## Testing

```bash
python3 -m pytest tests/ -q
```

---

## Data credit

Passenger records come from Kaggle’s
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
competition. This project is for education and portfolio demonstration, not a
competition submission package.

---

## Author

**Juan David Prada Malagón** — data scientist.

If this repo helps you learn logistic regression or structure an ML portfolio piece, a star is appreciated.
