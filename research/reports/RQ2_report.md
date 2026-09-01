# RQ2 — Do the External Factors Earn Their Place?

**Question:** How much forecast accuracy do weather, calendar and promotion
regressors contribute, and does the contribution differ by model family?

**Codefile:** `research/rq2_external_factors.py`
**Evaluation artifacts:** `research/results/rq2_global_ablation.csv`,
`rq2_local_regressor_value.csv`

## Method

**Global models (LightGBM, XGBoost):** leave-one-group-out ablation. Four variants
are trained per model — `full` (all 62 features), and one variant each with the
entire weather, calendar, or promotion column group removed (lag/rolling/price/id
columns are held constant in every variant, so a delta is attributable only to the
removed group). Same fit/val/test split as RQ1.

**Local model (SARIMA):** SARIMA has no regressor slots, so the comparison is
SARIMA (univariate, RQ1's number) vs. SARIMAX fit on the same 24-series sample with
six exogenous regressors — `is_weekend, is_public_holiday, is_school_holiday,
temp_anomaly, is_promotion, discount_pct` — one representative column per group,
the same set already used for Prophet's regressors in `04_modelling_update.ipynb`.

## Results

Global leave-one-group-out (full test set, 33,120 rows):

| model | variant | WAPE | MASE | WAPE cost of removal |
|---|---|---|---|---|
| LightGBM | full | 24.71% | 0.752 | — |
| LightGBM | no_weather | 25.15% | 0.766 | **+1.8%** |
| LightGBM | no_calendar | 27.31% | 0.831 | **+10.5%** |
| LightGBM | no_promotion | 25.06% | 0.763 | **+1.4%** |
| XGBoost | full | 26.41% | 0.804 | — |
| XGBoost | no_weather | 27.09% | 0.825 | +2.6% |
| XGBoost | no_calendar | 27.48% | 0.836 | +4.1% |
| XGBoost | no_promotion | 27.51% | 0.837 | +4.2% |

Local model:

| model | MAE | WAPE | MASE |
|---|---|---|---|
| SARIMA (no regressors) | 9.05 | 40.26% | 1.242 |
| SARIMAX (+regressors) | **7.90** | **35.14%** | **1.065** |

Adding the six regressors to the local model cuts its MASE by **14.2%**
(1.242 → 1.065) and its WAPE by 5.1 points.

## Interpretation

**Contribution differs sharply by model family, in the direction the codebase's
existing lag-absorption investigation (`06_proof_EXECUTED_example.ipynb`) predicts.**
SARIMA has no lag features at all, so it has nothing to fall back on when a
regressor is absent — adding those six regressors buys a 14% MASE improvement, far
larger than any single group's removal cost for the tree ensembles. The global
models, by contrast, already carry 20 lag/rolling/EWMA columns that encode recent
demand level; removing an *external* regressor group costs them only 1–10%. This
is consistent with `06_proof_EXECUTED_example.ipynb`'s H2 finding ("lag features
supply the demand level, which frees the exogenous features to explain deviations
from it") — external regressors matter far more to a model with no lag information
than to one that has it.

**Within the global family, the two models don't agree on which group matters
most.** Calendar features are worth far more to LightGBM (10.5% cost) than to
XGBoost (4.1% cost) — LightGBM apparently leans on calendar structure (day-of-week,
month, holiday proximity) more heavily than XGBoost does. Promotion features show
the opposite asymmetry: 4.2% cost for XGBoost vs. 1.4% for LightGBM. Weather is the
least valuable group for both (1.8% / 2.6%), consistent with the correlation
analysis from the EDA (temperature correlates only 0.02 with raw sales) and the
weak temperature-driver recovery already documented in the SHAP work.

## Evaluation notes / limitations

- **A real bug was caught and fixed during this evaluation.** The first run of
  SARIMAX-with-regressors silently failed to fit on all 24 series (a
  `statsmodels` date-index quirk when `exog` is a DataFrame sliced from a
  multi-series panel) and fell back to a naive 7-day mean for every series,
  which produced a false "regressors barely help" result (MAE 9.05 vs 9.15).
  Fixed in `common.run_sarimax` by resetting the index before fitting; the
  corrected run shows 0 fallbacks and a genuine, much larger effect. This is a
  useful reminder that a silent `except: fallback` block can hide a completely
  broken experimental arm — the fix now prints every fallback it takes.
- **Group definitions are a modelling choice.** `is_promotion`/`discount_pct` are
  simple, near-duplicate columns (r=0.90, see the earlier EDA), so the "promotion"
  group's contribution is really the contribution of one signal represented twice;
  a stricter ablation might drop only one.
- Local-model result is on the 24-series sample, not the full 360.
