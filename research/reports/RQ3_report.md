# RQ3 — Attribution Agreement and Recovery of Known Drivers

**Question:** Do SHAP attributions from LightGBM and XGBoost agree with each
other, with permutation importance, and with SARIMAX coefficients — and do they
recover the item-level drivers the generator actually used?

**Codefile:** `research/rq3_attribution_agreement.py`, `research/rq3_raw_outputs.py`,
`research/rq3_statistical_audit.py`
**Evaluation artifacts:** `research/results/rq3_attribution_agreement.csv`,
`rq3_sarimax_coefficients.csv`, `rq3_recovery.csv`, `rq3_raw_importances.csv`,
`rq3_item_mass_raw.csv`, `rq3_importance_shape.csv`, `rq3_same_footing_agreement.csv`,
`rq3_restricted_footing_agreement.csv`, `rq3_recovery_with_ci.csv`, `rq3_topk_recovery.csv`

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

### Statistical audit: is Spearman actually right here, are these methods comparable, and does the ranking survive recall/precision@k?

A principal-level audit (`ml-datasci/selecting-statistical-test` →
`ml-datasci/checking-test-assumptions` → `ml-datasci/reporting-effect-sizes`)
re-fit the models to recover the raw 62-feature SHAP/permutation vectors and
per-item attribution mass (`rq3_raw_outputs.py`), instead of trusting the
summary rho/p already on file.

**Step 1/2 — Spearman confirmed, and the raw vectors show why.** Shapiro-Wilk
on all four n=62 importance vectors rejects Normality decisively (p < 1e-13,
skew 3.9–6.2 — a handful of features carry most of the mass, the rest are
near zero). Pearson on the same vectors happens to agree *directionally* with
Spearman here (r=0.98 vs rho=0.96 for SHAP-LightGBM/XGBoost; r=0.96 vs
rho=0.92 for LightGBM SHAP-vs-permutation; r=0.96 vs rho=0.88 for XGBoost) —
unlike RQ2's weather correlation, there's no sign flip — but Pearson is
consistently **inflated** by 0.02–0.08 over Spearman at every pair, the
signature of a few extreme-importance features dominating a linear-covariance
measure. Spearman is still the correct choice; this is a case where the
audit *validates* the pre-existing method rather than overturning it.

**Step 2 — SHAP/permutation and SARIMAX are not on comparable footing, and the
original report never actually tested them against each other.** SARIMAX has
6 named exogenous regressors; SHAP/permutation cover 62 features, including
~20 lag/rolling/EWM columns SARIMAX structurally excludes (that's the entire
point of the lag-absorption finding this RQ builds on). A 62-feature agreement
test against SARIMAX is undefined. Restricting to the 6 shared regressor names
gives an answerable but severely underpowered comparison (`rq3_restricted_
footing_agreement.csv`, n=6): every pairing (SHAP or permutation, either
model, vs. SARIMAX |coefficient|) comes back **negative and non-significant**
(rho −0.09 to −0.43, p 0.40–1.0, Kendall tau −0.07 to −0.33). That is not
evidence of disagreement — n=6 is far below any reliable threshold for a rank
correlation, so no CI is reported — but it is also not evidence the original
report's implicit "these three methods triangulate" framing can lean on
either; the report never ran this comparison, and now that it has, there's
nothing there to cite.

A second, quieter footing mismatch: the item-level SARIMAX recovery numbers in
the table above are computed over **17 of the 30 items**, not 30 — the
24-series local sample doesn't cover every item — while both SHAP recovery
numbers use all 30. SARIMAX's 0.809 temperature rho and the SHAP models'
0.365/0.408 are not estimated on the same population of items.

**Step 3 — effect sizes with CI change the confidence, not the ranking**
(`rq3_recovery_with_ci.csv`):

| driver | method | n | rho | 95% CI |
|---|---|---|---|---|
| temperature | SHAP LightGBM | 30 | 0.365 | **[−0.007, 0.648]** |
| temperature | SHAP XGBoost | 30 | 0.408 | [0.041, 0.678] |
| temperature | SARIMAX coefficient | 17 | 0.809 | [0.478, 0.939] |
| price | SHAP LightGBM | 30 | 0.565 | [0.230, 0.781] |
| price | SHAP XGBoost | 30 | 0.527 | [0.182, 0.757] |
| price | SARIMAX coefficient | 17 | 0.333 | [−0.190, 0.708] |

SHAP-LightGBM's temperature recovery — reported in the original draft as
significant (p=0.047) — has a 95% CI that **crosses zero**. At p just under
0.05 with n=30, that's the exact fragility the codebase's own
`06_proof_EXECUTED_example.ipynb` already warns about ("the temperature
p-value on either side of 0.05 purely from a different explained sample") —
this is that warning materializing as an actual CI, not hypothetical. SARIMAX
temperature recovery and both price recovery numbers for SHAP are comfortably
CI-positive; SARIMAX price recovery's CI is wide and crosses zero, consistent
with the original "n.s." call.

**Step 4 — recall/precision@k, and it's a starker story than the correlations
suggested** (`rq3_topk_recovery.csv`):

| driver | method | recall/precision@5 | recall/precision@10 |
|---|---|---|---|
| temperature | SHAP LightGBM | **0.0** | 0.5 |
| temperature | SHAP XGBoost | **0.0** | 0.5 |
| temperature | SARIMAX coefficient | 0.4 | 0.7 |
| price | SHAP LightGBM | 0.2 | 0.7 |
| price | SHAP XGBoost | 0.4 | 0.7 |
| price | SARIMAX coefficient | 0.4 | 0.3 |

Both SHAP models get **zero of the true top-5 temperature-driven items right**
— a much sharper failure than a rho of 0.37–0.41 conveys on its own. SARIMAX
gets 2 of 5. At k=10 all three methods look more competitive for temperature,
but the top-5 result is the more decision-relevant one (identifying the
handful of items most worth a targeted intervention). For price, SARIMAX's
top-10 recall (0.3) is actually *worse* than its top-5 (0.4) — consistent with
its recovery rho being weak and non-significant — while both SHAP methods hold
up better at both k.

## Interpretation

**Agreement is high across the board — for the two methods that are actually
comparable.** SHAP rankings from LightGBM and XGBoost correlate at 0.96, and
each model's own SHAP ranking agrees with its permutation importance at
0.88–0.92, all with tight CIs (`rq3_same_footing_agreement.csv`). That's SHAP
and permutation importance, both scored over the same 62-feature universe —
attribution instability is not the concern for these two. **SARIMAX was never
actually part of that agreement claim.** It only has 6 exogenous regressors
against SHAP/permutation's 62, so there's no full-feature-set comparison to
run; restricted to the 6 shared regressor names (n=6), every SHAP/permutation-
vs-SARIMAX pairing comes back weakly negative and non-significant (statistical
audit above) — not evidence of disagreement, but not evidence for "three
methods agree" either. That framing should be retired in favor of "SHAP and
permutation importance agree with each other; SARIMAX addresses a different,
much narrower question and was never tested against them directly."

**Recovery of the true generator driver is a different question, and the answer is
driver-specific, not method-specific.** No single method recovers both drivers
best:

- **Temperature** is recovered far better by the **SARIMAX coefficient** (rho
  0.81, 95% CI [0.48, 0.94], n=17 items) than by either model's SHAP (0.37 on
  n=30 items — CI [−0.01, 0.65], crossing zero — /0.41, CI [0.04, 0.68]). The
  direction of this finding survives the audit, but two caveats now attach to
  it that didn't before: SARIMAX's number is estimated on 17 of the 30 items,
  not the 30 SHAP uses, so the two aren't strictly comparable populations; and
  SHAP-LightGBM's own recovery is CI-fragile, not a settled significant
  result. The **recall@5 numbers make the qualitative finding sharper anyway**
  — both SHAP models identify zero of the true top-5 temperature-driven items,
  versus SARIMAX's 2 of 5 — which is a more decision-relevant way to see the
  same gap than the correlation coefficients alone. This corroborates the "lag
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
