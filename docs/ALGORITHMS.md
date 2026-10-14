# Algorithms

## Custom logistic regression

Labels are mapped to `y_i ∈ {-1, +1}`. With feature vectors `x_i` (including an
intercept column) and weights `w`, the Bernoulli likelihood under the logistic
link is

$$
p(y_1,\ldots,y_n \mid x_1,\ldots,x_n, w)
  = \prod_{i=1}^{n} \sigma(y_i\, x_i^\top w)
$$

where

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The log-likelihood

$$
\mathcal{L}(w) = \sum_{i=1}^{n} \log \sigma(y_i\, x_i^\top w)
$$

has gradient

$$
\nabla_w \mathcal{L}
  = \sum_{i=1}^{n} \big(1 - \sigma(y_i\, x_i^\top w)\big)\, y_i\, x_i.
$$

Weights are updated by **batch gradient ascent**

$$
w \leftarrow w + \eta\, \nabla_w \mathcal{L},
$$

optionally with per-epoch averaging (`1/n`) and early stopping when the step
norm falls below a tolerance.

### Implementation upgrades vs. the original notebook

| Aspect | Original (2020) | Current |
|---|---|---|
| Gradient | Python `for` loop over passengers | Fully vectorized NumPy |
| Features | Raw columns after dummies | Engineered + scaled design matrix |
| Stopping | Fixed 12,000 iterations | Tolerance + max-iter |
| API | Loose functions | scikit-learn-compatible estimator |
| Evaluation | In-sample predictions | Stratified CV + holdout |

The custom model remains intentionally transparent: coefficients are readable,
training dynamics are inspectable via `history_`, and predictions share the same
interface as library classifiers.

## k-NN age imputation

Missing ages are predicted by a `KNeighborsRegressor` fit on passengers with
observed Age. Predictors are standardized so Fare does not dominate Euclidean
distance. Predicted ages are clipped to `[0, 80]`.

## Tree ensembles

Random Forests and Gradient Boosting capture non-additive effects (e.g.
`Sex × Pclass`) that a strictly linear logit cannot. They are included as
**performance anchors**: if they only marginally beat logistic regression, the
engineered linear features are already doing most of the work—an important
story for interpretable modeling.
