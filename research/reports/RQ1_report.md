# RQ1 — Model Family Comparison: Global ML vs Local Statistical

**Question:** Which of LightGBM, XGBoost and SARIMA achieves the best accuracy, and
under what series conditions does each win?

**Codefile:** `research/rq1_model_comparison.py`, `research/rq1_statistical_trio.py`,
`research/rq1_series_level_test.py`, `research/rq1_win_condition_check.py`
**Evaluation artifacts:** `research/results/rq1_overall.csv`, `rq1_segmented.csv`,
`rq1_dm_tests.csv`, `rq1_series_conditions.csv`, `rq1_statistical_trio.csv`,
`rq1_per_series_mase.csv`, `rq1_shapiro_differences.csv`, `rq1_omnibus_test.csv`,
`rq1_posthoc_wilcoxon.csv`, `rq1_gap_condition_correlation.csv`,
`rq1_gap_kruskal_wallis.csv`, `rq1_win_margin_effect_sizes.csv`

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

### Split audit and statistical trio on the accuracy deltas

Before trusting the table above, the split was audited for leakage
(`ml-datasci/auditing-train-test-split`): `common.time_split` holds out the last
28 training days for early stopping *inside* the training window, so early
stopping never touches the test set; SARIMA is fit per-series on `~is_test` and
forecasts strictly forward over the same cutoff; and the upstream feature
engineering (`03_feature_engineering_updated.ipynb`) fits quantile bin edges and
z-score normalisation on train rows only and builds all lag/rolling/EWM features
with `shift(1)` before rolling (a same-day cross-sectional leak in
`store_mean_7d`/`item_mean_7d` was caught and fixed there — see that notebook's
own note). No leakage found; the accuracy numbers above are trustworthy on that
front.

The three pairwise comparisons above (`rq1_dm_tests.csv`) were then run through
the rest of the statistical trio (`ml-datasci/comparing-models-fairly`,
`research/rq1_statistical_trio.py` → `rq1_statistical_trio.csv`):

| pair | DM stat | p-value | MSE delta (A−B) | 95% CI | WAPE delta (A−B, pp) | Holm α | significant |
|---|---|---|---|---|---|---|---|
| LightGBM vs XGBoost | −5.32 | 1.0e-7 | −7.67 | [−10.50, −4.85] | −1.85 | 0.0500 | yes |
| LightGBM vs SARIMA | −15.98 | ~0 | −100.31 | [−112.62, −88.01] | −16.79 | 0.0167 | yes |
| XGBoost vs SARIMA | −15.63 | ~0 | −92.64 | [−104.26, −81.02] | −14.93 | 0.0250 | yes |

DM (on paired squared-error loss, same 2,208 test rows) is the right test here —
it's the point-forecast analogue of a paired t-test on per-instance error, exactly
what `comparing-models-fairly` recommends for same-instance regression
comparisons. The effect-size CI is not a bootstrap: the per-row prediction frame
isn't persisted, so the CI is instead recovered analytically from the same
moments the DM statistic already uses (`d_bar = MSE_a − MSE_b` from the RMSE
column, `SE` backed out of `dm_stat = d_bar / SE`) — an exact companion to the DM
test rather than a separate approximation. With `m = 3` pairwise comparisons,
Holm-Bonferroni correction is required and applied; **all three differences
survive correction**, so LightGBM > XGBoost > SARIMA is not a multiple-comparisons
artifact. All three MSE-delta CIs also exclude zero by a wide margin, and the
WAPE deltas (1.85–16.79 pp) comfortably clear any reasonable operational-
significance bar for this use case — though note no such bar was pre-registered
for this RQ, so that judgment is informal.

### Stricter re-audit: the DM test's n=2,208 is pseudo-replication — the real unit is 24 series

A follow-up principal-level audit (`ml-datasci/auditing-train-test-split` →
`ml-datasci/selecting-statistical-test` → `ml-datasci/checking-test-assumptions` →
`ml-datasci/reporting-effect-sizes`, in that order) re-examined the trio above.

**Step 1 (split audit) passes** — nothing new: `common.time_split` never leaks
test rows into fit or early-stopping, SARIMA forecasts strictly forward, and
`store_item` legitimately repeating across train/test is correct here (a
forecasting task needs a series' own history in train), not group-leakage. One
scope note carries forward: this is a **single-origin** split (one fixed cutoff),
not multi-origin walk-forward CV, so the 24 sampled series — not repeated time
folds — are the actual unit of replication.

**Step 2 (test selection) is where the original trio has a real problem.** The
design is 3 models scored on the **same 24 series** — a repeated-measures /
paired design — but the DM test pools all 24 series' test-window rows into
n=2,208 and treats them as i.i.d. This double-counts: rows within a series are
serially autocorrelated (today's forecast error correlates with tomorrow's for
the same series), and the `diebold_mariano()` variance estimator in `common.py`
uses a plain `np.var(d)/n` with no HAC/Newey-West correction for that
autocorrelation. The DM p-values above are very likely **overconfident** — the
true effective sample size is closer to 24 (series) than 2,208 (rows). The
correct design-matched test is a repeated-measures comparison across the 24
series, gated by a Normality check on the per-series differences (Demsar 2006).

**Step 3 (assumption check) rules out the parametric option.** Refitting the
three models to recover the actual per-series MASE table (`rq1_per_series_mase.csv`,
via `research/rq1_series_level_test.py`) and running Shapiro-Wilk on the three
pairwise per-series differences:

| pair | n | Shapiro W | p | Normal? |
|---|---|---|---|---|
| LightGBM − XGBoost | 24 | 0.872 | 0.0057 | No |
| LightGBM − SARIMA | 24 | 0.554 | ~0 | No |
| XGBoost − SARIMA | 24 | 0.592 | ~0 | No |

All three reject Normality, which rules out repeated-measures ANOVA and gates
the choice to the **Friedman test** (omnibus) with **Wilcoxon signed-rank
post-hoc, Holm-Bonferroni corrected**. Friedman: χ² = 46.08, p = 9.8e-11 — the
three models are not interchangeable. Post-hoc (`rq1_posthoc_wilcoxon.csv`):

| pair | Wilcoxon stat | p (raw) | median MASE delta | Holm-significant |
|---|---|---|---|---|
| LightGBM vs XGBoost | 0.0 | 1.2e-7 | −0.062 | yes |
| LightGBM vs SARIMA | 0.0 | 1.2e-7 | −0.410 | yes |
| XGBoost vs SARIMA | 3.0 | 6.0e-7 | −0.339 | yes |

The qualitative ranking (LightGBM > XGBoost > SARIMA) **survives** the
design-correct test — but it now rests on the right n (24 series) instead of an
inflated one, which is what a principal reviewer would have blocked the original
trio on before it went further.

**Step 4 (effect size): report win-margin, not a bare "X wins" count.**
Matched-pairs rank-biserial r, bootstrap 95% CI over the 24 series
(`rq1_win_margin_effect_sizes.csv`):

| pair | rank-biserial r | 95% CI | direction |
|---|---|---|---|
| LightGBM vs XGBoost | 1.00 | [1.00, 1.00] | LightGBM has lower MASE in 24/24 series |
| LightGBM vs SARIMA | 1.00 | [1.00, 1.00] | LightGBM has lower MASE in 24/24 series |
| XGBoost vs SARIMA | 0.98 | [0.90, 1.00] | XGBoost has lower MASE in 23/24 series |

r ≈ 1 is about as large as this effect size gets — LightGBM's advantage over
both rivals is not a marginal, sampling-noise-adjacent win.

**Step 5: which series conditions actually support a win claim — corrected.**
The original draft framed this RQ as "under what series conditions does each
model win," which presupposes the winner can flip. It doesn't, in this sample:
rank-biserial r = 1.00 means LightGBM has zero losses across all 24 series
regardless of volume, volatility, or sparsity — there is no series condition in
this dataset under which SARIMA or XGBoost outright wins against LightGBM.
The one partial exception is `store_item 12_30` (low volume, high volatility),
the single series where SARIMA (0.900 MASE) edges out XGBoost (1.026) — but
LightGBM still wins there too (0.796). So the honest question isn't "which
conditions flip the winner" (none do), it's "which conditions predict the
*margin*" — and that's where the original hypothesis (`rq1_win_condition_check.py`)
partially survives and partially doesn't:

| condition tested | result | survives? |
|---|---|---|
| SARIMA-LightGBM gap vs. `zero_rate` (Spearman) | rho = −0.45, 95% CI [−0.73, −0.03], p = 0.028 | **Yes** — sparser series have a smaller LightGBM advantage |
| SARIMA-LightGBM gap across `volume_band` (Kruskal-Wallis) | H = 12.02, p = 0.0025 | **Yes** — corroborates the volume-band pattern already in the segmented table |
| SARIMA-LightGBM gap vs. `mean_volume` (Spearman) | rho = +0.41, 95% CI [−0.01, +0.71], p = 0.047 | **Borderline** — CI touches zero at n=24; treat as suggestive, not confirmed (the codebase's own `06_proof_EXECUTED_example.ipynb` flags exactly this fragility for small-n Spearman correlations) |
| SARIMA-LightGBM gap vs. `cv` / `cv_band` (Spearman rho = −0.34, p = 0.11; Kruskal-Wallis H = 3.20, p = 0.07) | not significant | **No** — the original report's volatility-band narrative is not statistically distinguishable from noise at n=24 |

**What forced the change:** step 2 (test selection) is what forces the headline
correction — the original "LightGBM wins all 24/24, gap narrows on low-volume
and widens on high-volatility series" claim was descriptively true but rested on
band-mean eyeballing with no test attached. Step 4 (effect-size + CI discipline)
is what forces the *volatility* piece of that claim to be dropped: once a real
test with a CI is run on it, it doesn't clear significance. The *sparsity*
(`zero_rate`) relationship is a strengthening, not present in the original
report's volume/cv framing at all, and it's the cleanest surviving signal —
SARIMA's relative disadvantage shrinks specifically on intermittent-demand
series, not generically on "low volume" ones.

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
