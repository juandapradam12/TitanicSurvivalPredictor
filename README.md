# Titanic Survival Predictor

**Interpretable binary classification with a from-scratch logistic regression core — plus leak-safe CV, tuning, calibration, SHAP, a Kaggle submission path, and a Streamlit demo.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/data-Kaggle%20Titanic-lightgrey.svg)](https://www.kaggle.com/c/titanic/data)

---

### Why this project exists

Most Titanic tutorials call `LogisticRegression().fit(...)` and stop. This repository goes further:

1. **Derive and implement** logistic regression from the likelihood and its gradient.
2. **Impute missing ages inside each CV fold** (scaled k-NN or group medians) — no leakage.
3. **Engineer features** that encode social structure (title, family size, cabin presence).
4. **Compare models fairly** with stratified cross-validation, then **explain** them with coefficients + SHAP.
5. **Ship a Kaggle submission** with one command.

The result is a portfolio piece that shows *statistical understanding*, *clean engineering*, and *honest evaluation*.

---

## Highlights

| Capability | Detail |
|---|---|
| Custom MLE logistic regression | Vectorized gradient ascent, scikit-learn-compatible API |
| Leak-safe Pipelines | Feature build → age impute → encode/scale nested in CV |
| Feature engineering | Title, FamilySize, IsAlone, HasCabin, LogFare |
| Model zoo | Custom LR · sklearn LR · Random Forest · Gradient Boosting |
| Evaluation | Stratified 5-fold CV (Accuracy / F1 / ROC-AUC) + holdout |
| Kaggle submission | `scripts/make_submission.py` → `submission.csv` |
| Interpretation | Coefficient tables + SHAP mean \|SHAP\| rankings |
| Hyperparameter tuning | RandomizedSearchCV over logistic / GBM / RF Pipelines |
| Calibration | Isotonic/Platt + F1-optimal decision threshold |
| Interactive demo | `streamlit run app.py` |
| CI | GitHub Actions: pytest + smoke benchmark |
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

# Benchmark (leak-safe CV)
python3 scripts/train_and_evaluate.py

# Kaggle submission (auto-picks best CV model)
python3 scripts/make_submission.py --model auto

# Coefficients + SHAP figures
python3 scripts/interpret_model.py --model sklearn_logistic
python3 scripts/interpret_model.py --model gradient_boosting

# Tune + calibrate (optional but recommended before serving)
python3 scripts/tune_model.py --model gradient_boosting --n-iter 20
python3 scripts/calibrate_and_threshold.py --model gradient_boosting

# Interactive demo
streamlit run app.py

# Notebook
jupyter notebook TitanicSurvivalPredictor.ipynb
```

On the engineered feature set with **fold-safe preprocessing**, stratified 5-fold CV typically lands around **~83–84% accuracy** and **~0.88 ROC-AUC**. Gradient Boosting usually leads; the **custom logistic regression stays within ~1 point of scikit-learn’s**.

---

## Project layout

```text
TitanicSurvivalPredictor/
├── TitanicSurvivalPredictor.ipynb
├── train.csv / test.csv
├── requirements.txt
├── src/titanic_predictor/
│   ├── custom_logistic.py      # From-scratch logistic regression
│   ├── preprocessing.py        # Leak-safe feature / impute / encode Pipelines
│   ├── features.py             # Feature helpers
│   ├── imputation.py           # Classic k-NN helpers
│   ├── models.py               # Model zoo + CV
│   ├── interpretation.py       # Coefficients + SHAP
│   └── pipeline.py             # Benchmark, submit, interpret entry points
├── app.py                           # Streamlit demo
├── scripts/
│   ├── train_and_evaluate.py
│   ├── make_submission.py
│   ├── interpret_model.py
│   ├── tune_model.py
│   └── calibrate_and_threshold.py
├── .github/workflows/ci.yml
├── tests/
├── figures/
└── docs/
    ├── METHODOLOGY.md
    ├── ALGORITHMS.md
    ├── INTERPRETATION.md
    ├── TUNING_AND_CALIBRATION.md
    └── PROJECT_OVERVIEW.md
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
Samples: 891  |  Features: 17
Best model (by ROC-AUC CV): gradient_boosting

            model  accuracy_mean  f1_mean  roc_auc_mean
gradient_boosting         0.8406   0.7785        0.8816
    random_forest         0.8328   0.7687        0.8775
 sklearn_logistic         0.8283   0.7729        0.8768
  custom_logistic         0.8283   0.7711        0.8759
```

### What drives survival?

<p align="center">
  <img src="figures/coefficients_sklearn_logistic.png" alt="Logistic coefficients" width="720"/>
</p>

<p align="center">
  <img src="figures/shap_importance_gradient_boosting.png" alt="SHAP importance for Gradient Boosting" width="720"/>
</p>

Title, sex, and class dominate both the linear coefficients and the tree SHAP rankings — the story an interpretable model should recover.

### Calibration & ROC

<p align="center">
  <img src="figures/reliability_gradient_boosting.png" alt="Reliability curve" width="420"/>
  <img src="figures/roc_pr_gradient_boosting.png" alt="ROC and PR curves" width="520"/>
</p>

Tune with `scripts/tune_model.py`, then calibrate and pick an F1-optimal threshold with `scripts/calibrate_and_threshold.py`. Serve interactively via `streamlit run app.py`.

---

## Documentation

- [Methodology](docs/METHODOLOGY.md) — leak-safe Pipelines, features, CV
- [Algorithms](docs/ALGORITHMS.md) — custom logistic regression & imputation
- [Interpretation](docs/INTERPRETATION.md) — coefficients & SHAP
- [Tuning & calibration](docs/TUNING_AND_CALIBRATION.md) — search spaces, Brier, thresholds
- [Project overview](docs/PROJECT_OVERVIEW.md) — elevator pitch & design principles
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
competition. `train.csv` and `test.csv` are included so the submission script runs out of the box.

---

## Author

**Juan David Prada Malagón** — data scientist.

If this repo helps you learn logistic regression or structure an ML portfolio piece, a star is appreciated.
