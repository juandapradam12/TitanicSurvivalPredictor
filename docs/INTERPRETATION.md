# Interpretation guide

This project treats interpretability as a first-class deliverable — not an
afterthought after chasing leaderboard points.

## Logistic coefficients

For `custom_logistic` and `sklearn_logistic`, each coefficient is the change in
**log-odds** of survival associated with a one-unit increase in that feature
(after preprocessing). The odds ratio is `exp(coef)`.

Generate the table and bar chart:

```bash
python3 scripts/interpret_model.py --model sklearn_logistic
```

Typical strongest drivers on this dataset:

| Signal | Direction | Intuition |
|---|---|---|
| `Title_Master` | ↑ survival | Young boys traveling with family |
| `Title_Mr` / `Sex_male` | ↓ survival | Adult men had lower priority in evacuation |
| `Title_Mrs` | ↑ survival | Women (often with children) |
| `Pclass` | ↓ as class number ↑ | Higher socioeconomic status → better access |

## SHAP values

SHAP attributes a prediction to features by averaging marginal contributions
across coalitions. The CLI writes **mean |SHAP|** rankings:

```bash
python3 scripts/interpret_model.py --model gradient_boosting
python3 scripts/interpret_model.py --model sklearn_logistic
```

- Linear models use `shap.LinearExplainer`
- Tree ensembles use `shap.TreeExplainer`

Agreement between coefficient magnitude and mean |SHAP| is a useful sanity check:
title / sex / class should dominate both views.

## Programmatic API

```python
from titanic_predictor.pipeline import train_final_model, interpret_model

model, X, y = train_final_model(model_name="sklearn_logistic")
bundle = interpret_model(model, X)
print(bundle["coefficient_narrative"])
print(bundle["shap_importance"].head())
```
