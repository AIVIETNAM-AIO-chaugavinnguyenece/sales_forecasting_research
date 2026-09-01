# RQ3 — Attribution Agreement and Recovery of Known Drivers

**Question:** Do SHAP attributions from LightGBM and XGBoost agree with each
other, with permutation importance, and with SARIMAX coefficients — and do they
recover the item-level drivers the generator actually used?

**Codefile:** `research/rq3_attribution_agreement.py`
**Evaluation artifacts:** `research/results/rq3_attribution_agreement.csv`,
`rq3_sarimax_coefficients.csv`, `rq3_recovery.csv`

## Method

SHAP importance uses `shap.TreeExplainer` in interventional mode with an
independent masker (800 explained rows, 60 background rows, identical seeded
sample for both models — asserted equal at runtime). Both LightGBM and XGBoost are
refit on a numeric-coded feature matrix with `enable_categorical=False`, not the
native-categorical models from RQ1/RQ2: SHAP's interventional `TreeExplainer`
raises `NotImplementedError` on XGBoost's native categorical splits, so this study
follows the existing precedent in `06_proof_EXECUTED_example.ipynb`, which hit the
same constraint and used the same workaround. Permutation importance uses
`sklearn.inspection.permutation_importance` (`neg_mean_absolute_error`, 5 repeats,
3,000-row sample). SARIMAX coefficients come from the same 24-series/6-regressor
fit as RQ2, with `exog` passed as a **named** DataFrame so `fit.param_names` gives
the true regressor names — extracting by name rather than position, since
parameter order shifts with the trend/seasonal spec and a positional slice would
silently return the wrong coefficient. Recovery is Spearman rho between each
method's attribution mass and the generator's held-out `abs_temp_effect` /
`abs_elasticity` per item (`common.ITEM_TRUTH`), never given to any model.

## Results

**Cross-method agreement:**

| comparison | rho | p-value |
|---|---|---|
| SHAP LightGBM vs SHAP XGBoost | **0.963** | ~0 |
| LightGBM: SHAP vs permutation | 0.917 | — |
| XGBoost: SHAP vs permutation | 0.883 | — |

**Driver recovery (Spearman rho vs. generator ground truth):**

| driver | method | rho | p-value |
|---|---|---|---|
| temperature | SHAP LightGBM | 0.365 | 0.047 |
| temperature | SHAP XGBoost | 0.408 | 0.025 |
| temperature | **SARIMAX coefficient** | **0.809** | **0.0001** |
| price | SHAP LightGBM | **0.566** | **0.001** |
| price | SHAP XGBoost | **0.527** | **0.003** |
| price | SARIMAX coefficient | 0.333 | 0.192 (n.s.) |

## Interpretation

**Agreement is high across the board.** SHAP rankings from LightGBM and XGBoost
correlate at 0.96, and each model's own SHAP ranking agrees with its permutation
importance at 0.88–0.92. Three independent attribution methods substantially agree
on *which* features matter for the global models — attribution instability is not
the concern here.

**Recovery of the true generator driver is a different question, and the answer is
driver-specific, not method-specific.** No single method recovers both drivers
best:

- **Temperature** is recovered far better by the **SARIMAX coefficient** (rho
  0.81) than by either model's SHAP (0.37/0.41). This corroborates the "lag
  absorption" finding already established in `06_proof_EXECUTED_example.ipynb`:
  SARIMAX has no lag/rolling features to route the temperature response through,
  so its `temp_anomaly` coefficient has to carry that signal directly, while
  LightGBM/XGBoost can (and evidently do) let their 20 lag/rolling columns absorb
  part of it. The SHAP LightGBM/XGBoost values here (0.365/0.408) closely match
  the "with_lags" arm's values in that earlier notebook (0.378/0.408) — a
  consistency check that this reimplementation reproduces the established result.
- **Price** flips the ranking: SHAP recovers it well (0.53–0.57) but the SARIMAX
  coefficient does not (0.33, not significant). The `rq3_sarimax_coefficients.csv`
  table shows why — `discount_pct` coefficients are unstable and even sign-flip
  across stores selling the same item (e.g. item 3: +0.26 at store 28, −0.44 and
  −0.13 at two other stores). `discount_pct` and `is_promotion` are correlated at
  r=0.90 (see the EDA), and with both in the same 6-regressor exogenous set,
  collinearity likely makes the discount coefficient a noisy read on an item's true
  elasticity, whereas SHAP's mass-based attribution isn't thrown by two correlated
  regressors the same way a single coefficient estimate is.

**Bottom line:** none of the three methods is uniformly the most faithful. SHAP is
the more trustworthy read on price elasticity here; a lag-free local model's
coefficient is the more trustworthy read on the temperature response. A paper
claiming "SHAP recovers the generator's drivers" needs that caveat attached
per-driver, not as a blanket statement.

## Evaluation notes / limitations

- The attribution-study models (300 trees, numeric-coded features) are **not**
  the same model objects RQ1/RQ2 forecast with (2000 trees, native categorical
  handling) — this mirrors the codebase's own established practice, but means
  RQ3's accuracy is not directly comparable to RQ1's.
- SARIMAX coefficients are aggregated to item level by averaging across the 1–3
  stores per item present in the 24-series sample; some items have only one
  observation, so the coefficient-based recovery rho is sensitive to a small
  number of series. Re-running with `N_LOCAL_SERIES = None` (all 360 series, all
  30 items with multiple stores each) would give a materially more stable
  estimate of the SARIMAX recovery numbers specifically.
- `p_value` is left blank for the two SHAP-vs-permutation rows in
  `rq3_attribution_agreement.csv` deliberately — permutation importance across 62
  features doesn't have a single well-defined p-value against a SHAP ranking the
  way a fresh Spearman test does for the fully independent recovery comparisons;
  only the rho is reported for those two rows.
