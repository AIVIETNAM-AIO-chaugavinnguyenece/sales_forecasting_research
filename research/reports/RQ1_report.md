# RQ1 — Model Family Comparison: Global ML vs Local Statistical

**Question:** Which of LightGBM, XGBoost and SARIMA achieves the best accuracy, and
under what series conditions does each win?

**Codefile:** `research/rq1_model_comparison.py`
**Evaluation artifacts:** `research/results/rq1_overall.csv`, `rq1_segmented.csv`,
`rq1_dm_tests.csv`, `rq1_series_conditions.csv`

## Method

LightGBM and XGBoost are trained once as global models across all 360 series (the
last 28 training days held out for early stopping, never the test set — see
`common.time_split`). SARIMA(1,1,1)(1,0,1,7), univariate, is fit per series on a
24-series sample drawn evenly across the volume range (`common.select_local_series`).
All three are then scored on exactly those 24 series over the same October–December
2025 test window, which is the only way the global and local families are
comparable. MASE is the headline metric because it scales each series by its own
in-sample naive error, so it isn't dominated by high-volume series the way MAE/RMSE
are.

## Results

Overall (24-series subset):

| model | MAE | RMSE | WAPE | MASE |
|---|---|---|---|---|
| LightGBM | 5.28 | 7.93 | 23.47% | **0.724** |
| XGBoost | 5.69 | 8.40 | 25.32% | 0.790 |
| SARIMA | 9.05 | 12.78 | 40.26% | 1.242 |

Diebold-Mariano on squared-error loss (all pairs, n=2,208): every pair differs
significantly (p < 0.0001) — LightGBM vs XGBoost (stat −5.32), LightGBM vs SARIMA
(stat −15.98), XGBoost vs SARIMA (stat −15.63). None of the differences are
sampling noise.

Segmented MASE:

| segment | LightGBM | XGBoost | SARIMA |
|---|---|---|---|
| low volume | 0.696 | 0.801 | 0.957 |
| mid volume | 0.735 | 0.771 | 1.580 |
| high volume | 0.742 | 0.799 | 1.187 |
| low volatility (CV) | 0.696 | 0.744 | 1.376 |
| high volatility (CV) | 0.753 | 0.837 | 1.107 |

Per-series winner (by MASE): **LightGBM wins all 24 of 24 sampled series.** SARIMA
never wins outright in this sample, but its gap to LightGBM is narrowest on
low-volume series (0.957 vs 0.696, a 0.26 gap) and widest on mid-volume series
(1.580 vs 0.735, a 0.85 gap) — SARIMA is comparatively least uncompetitive on the
sparsest series, where a global model's cross-series pooling advantage is
presumably smallest relative to a model fit purely on that series' own history.

## Interpretation

The global tree ensembles dominate SARIMA outright in this sample — there is no
series condition under which SARIMA wins, only conditions under which it loses by
less. This is consistent with the "global model borrows strength across series"
framing in `04_modelling_update.ipynb`: SARIMA sees only ~2.4 years of one series'
own history and no exogenous information, while LightGBM/XGBoost pool 360 series'
worth of patterns and (implicitly, via lag/rolling features) each series' recent
level. LightGBM is consistently the best global model, and this ranking is stable
under Diebold-Mariano significance even though XGBoost's exact score is not
run-to-run deterministic (below).

## Evaluation notes / limitations

- **XGBoost non-determinism:** with `tree_method="hist"` and `n_jobs=-1`, XGBoost's
  histogram construction is not bit-reproducible across runs even with a fixed
  `random_state` — a second run of this script produced XGBoost MAE 5.69 versus
  5.39 in the codebase's existing `04_modelling_update.ipynb`. LightGBM's result
  (MAE 5.2760) matched the existing notebook exactly. This does not change the
  qualitative ranking (LightGBM ≥ XGBoost ≫ SARIMA held in both runs), but any
  claim more precise than "LightGBM and XGBoost are close, both clearly beat
  SARIMA" should be re-verified with `n_jobs=1` for a fully reproducible run.
- **Sample size:** the local-model comparison uses 24 of 360 series
  (`N_LOCAL_SERIES` in `common.py`). The "SARIMA never wins" finding is based on
  this sample; set `N_LOCAL_SERIES = None` in `common.py` and re-run for the full
  360-series population before treating it as a population-level claim.
- **ARIMA and Prophet are intentionally excluded** per this RQ's scope (SARIMA
  only, per the user's phrasing), unlike the original 5-model comparison in
  `04_modelling_update.ipynb`.
