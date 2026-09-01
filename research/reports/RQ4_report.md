# RQ4 — Normalisation and Target Transformation

**Question:** How do target transformation and cross-series normalisation affect
forecast accuracy, and does the effect differ systematically between global tree
ensembles and local statistical models?

**Codefile:** `research/rq4_normalization_transform.py`
**Evaluation artifacts:** `research/results/rq4_global_transforms.csv`,
`rq4_local_transforms.csv`

## Pre-registered predictions

Fixed in the script's docstring before any number below was computed:

- **P1** log1p changes global-model WAPE by more than it changes SARIMA's WAPE.
- **P2** per-series z-score normalisation meaningfully helps at least one global model.
- **P3** falsification — if both families move by a similar amount, the effect is generic.

## Method

Global models: three target variants on the full training set — `raw`, `log1p`
(fit on `log1p(sales)`, inverted with `expm1` before scoring), `per_series_zscore`
(fit on `(sales − series_train_mean) / series_train_std`, inverted before
scoring; std of 0 clipped to 1). Local model: SARIMA on the 24-series sample,
`raw` vs `log1p` — per-series z-score is not tested for SARIMA, because linear
rescaling of a linear Gaussian ARMA model's target does not change its forecasts
once inverted, so it isn't an interesting comparison for that family.

## Results

| model | variant | WAPE | MASE |
|---|---|---|---|
| LightGBM | raw | 24.71% | 0.752 |
| LightGBM | log1p | **24.47%** | 0.745 |
| LightGBM | per_series_zscore | 24.72% | 0.753 |
| XGBoost | raw | 26.41% | 0.804 |
| XGBoost | log1p | 27.92% | 0.850 |
| XGBoost | per_series_zscore | 28.75% | 0.875 |
| SARIMA | raw | 40.26% | 1.242 |
| SARIMA | log1p | **37.57%** | **1.156** |

WAPE improvement from baseline (positive = better):

| | log1p | per-series z-score |
|---|---|---|
| LightGBM | +0.98% | −0.05% (no effect) |
| XGBoost | **−5.72%** (worse) | **−8.85%** (worse) |
| SARIMA | **+6.69%** | not tested |

**P1: False.** SARIMA moved the most under log1p (+6.7% WAPE), more than either
global model. The prediction had the direction backwards.
**P2: False.** Per-series z-score normalisation helps neither global model —
it's flat for LightGBM and actively harmful for XGBoost.
**P3: False.** The effect sizes don't converge to a common value either — they
range from −8.85% to +6.69%, and the sign itself differs across model/variant
combinations, which is a family-specific pattern, just not the one predicted.

## Interpretation

The falsification is informative rather than a dead end. **Target transformation
matters most for the model that has no other way to know a series' scale.** SARIMA
sees only its own series' raw values — no item/store identifiers, no lag features
— so it has no substitute for a well-behaved target distribution, and the
generator's Poisson-like multiplicative noise (documented in `data/README.md`)
is exactly the kind of variance-scales-with-level pattern log1p is designed to
stabilize. The global tree ensembles, by contrast, already receive 20 lag/rolling
columns (`sales_lag_1`, `sales_mean_7d`, `store_mean_7d`, `item_mean_7d`, etc.)
plus categorical `store_id`/`item_id` — these features already tell the model
which scale a given row belongs to, so transforming the *target* on top of that is
redundant at best. That also explains why per-series z-score normalisation (which
is only useful for helping a model discover which scale a row belongs to) does
nothing for LightGBM and actively hurts XGBoost: the information the transform
would supply is already present as input features, so the transform only adds
optimization noise. XGBoost is the more sensitive of the two to any target
transform, degrading under both log1p and z-score, plausibly because its
early-stopping validation window doesn't tune as robustly on a differently-scaled
loss surface as LightGBM's does (LightGBM slightly improves under log1p, so the
harm isn't inherent to changing the target — it's model-specific).

**Answer to the RQ:** yes, the effect differs systematically by family, but the
direction is the reverse of the "global models need normalisation to pool
heterogeneous series" intuition this study started with. It's the **lag-feature-free
local model** that benefits from target transformation; the global models, already
carrying explicit scale information via lag/rolling/identifier features, see no
benefit and sometimes real harm.

## Evaluation notes / limitations

- This connects to the same mechanism documented in
  `06_proof_EXECUTED_example.ipynb` and reflected in RQ2's results: lag/rolling
  features carry information that other techniques (regressors in RQ2, target
  transforms here) become mostly redundant with once they're present. A model
  without them (SARIMA) benefits far more from anything that supplies that
  missing information, however it's supplied.
- XGBoost's degradation under both transforms should be treated with some caution
  given the run-to-run non-determinism documented in the RQ1 report
  (`tree_method="hist"`, `n_jobs=-1`); the direction (transforms hurt XGBoost more
  than LightGBM) is large enough here (5–9 points of WAPE) that it's very unlikely
  to be fully explained by that noise, but an `n_jobs=1` re-run would confirm.
- Local-model result is on the 24-series sample, not the full 360.
