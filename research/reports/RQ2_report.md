# RQ2 — Do the External Factors Earn Their Place?

**Question:** How much forecast accuracy do weather, calendar and promotion
regressors contribute, and does the contribution differ by model family?

**Codefile:** `research/rq2_external_factors.py`, `research/rq2_weather_correlation_audit.py`,
`research/rq2_series_level_test.py`, `research/rq2_statistical_audit.py`
**Evaluation artifacts:** `research/results/rq2_global_ablation.csv`,
`rq2_local_regressor_value.csv`, `rq2_weather_correlation.csv`,
`rq2_weather_correlation_item_level.csv`, `rq2_global_per_series_mase.csv`,
`rq2_local_per_series_mase.csv`, `rq2_main_effect_tests.csv`,
`rq2_interaction_tests.csv`, `rq2_local_test.csv`

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

### Statistical audit: is the group contribution real, and is the model-family interaction real, or fold noise?

The table above is n=1 per (model, variant): one WAPE number over the pooled
33,120-row test set, with no replicate to attach a p-value or CI to. A
principal-level audit (`ml-datasci/selecting-statistical-test` →
`ml-datasci/checking-test-assumptions` → `ml-datasci/reporting-effect-sizes`)
re-fit the same 8 (model × variant) combinations but recovered **per-series**
MASE (`rq2_series_level_test.py` → `rq2_global_per_series_mase.csv`, 360
series), which turns "with vs without" into a proper paired design and makes
the model-family question an answerable **interaction** test instead of an
eyeballed comparison of two WAPE-cost percentages.

**Step 1 (test selection).** Two different questions need two different tests,
and neither is independent-samples:
- *Does group G help model M?* — paired (same 360 series, with vs without G).
- *Does G's contribution differ between LightGBM and XGBoost?* — an
  **interaction**, tested as a difference-of-differences per series
  (`delta_LightGBM_i = MASE(no_G)_i − MASE(full)_i` vs. the same for XGBoost,
  paired because both deltas share the same series), not a comparison of two
  aggregate WAPE-cost numbers.

**Step 2 (assumption check).** Shapiro-Wilk on each of the 9 paired-difference
vectors (6 main-effect + 3 interaction) rejected Normality in 8 of 9 — Wilcoxon
signed-rank is the correct test in those cases. The one exception
(promotion's interaction contrast, Shapiro p = 0.14) is Normal, so that one
comparison correctly used a paired t-test / Cohen's dz instead — the gate did
real branching, not a rubber-stamp.

**Main effect (with vs. without), per model family** (`rq2_main_effect_tests.csv`,
Holm-Bonferroni across the 6 tests, all significant):

| model | group | median MASE cost of removal | rank-biserial r | 95% CI |
|---|---|---|---|---|
| LightGBM | weather | +0.011 | 0.76 | [0.69, 0.83] |
| LightGBM | calendar | +0.071 | **1.00** | [0.99, 1.00] |
| LightGBM | promotion | +0.008 | 0.45 | [0.34, 0.55] |
| XGBoost | weather | +0.021 | 0.91 | [0.87, 0.95] |
| XGBoost | calendar | +0.028 | 0.56 | [0.46, 0.65] |
| XGBoost | promotion | +0.036 | 0.93 | [0.89, 0.96] |

Every group helps every model — none of the six is noise — but the *size* of
that help is nowhere near uniform, which is exactly why a pooled "external
factors help" number would have hidden the real story (per this RQ's step 3
requirement). One correction to the original draft's own framing: **"weather
is the least valuable group for both" is not quite right for LightGBM** —
LightGBM's promotion cost (+0.008) is smaller than its weather cost (+0.011),
so promotion, not weather, is LightGBM's least valuable group. This matches
the original WAPE-cost numbers too (1.4% promotion < 1.8% weather for
LightGBM) — the inaccuracy was already latent in the original table, just not
stated plainly.

**Interaction (does the group's contribution differ by model family?)**
(`rq2_interaction_tests.csv`, Holm-Bonferroni across the 3 tests, all
significant):

| group | test | median (LightGBM delta − XGBoost delta) | effect | 95% CI |
|---|---|---|---|---|
| weather | Wilcoxon | −0.008 | r = 0.46 | [0.36, 0.57] |
| calendar | Wilcoxon | +0.049 | r = **0.97** | [0.96, 0.99] |
| promotion | paired t-test | −0.028 | dz = **−0.98** | [−1.11, −0.86] |

**Verdict: the interaction is real, not fold noise.** All three regressor
groups show a statistically significant, Holm-corrected, medium-to-huge
model-family interaction: XGBoost depends on weather and promotion more than
LightGBM does; LightGBM depends on calendar far more than XGBoost does (r=0.97
is close to the ceiling — almost every one of the 360 series shows this same
direction). This confirms — with a real test and effect size behind it — the
original draft's claim that "the two models don't agree on which group matters
most." Nothing here forced a reversal of that conclusion; step 1–2 forced the
*method* (paired difference-of-differences, not two aggregate percentages
eyeballed against each other), and the result came out the same direction the
draft already had.

**Local model** (`rq2_local_test.csv`): SARIMA vs. SARIMAX paired over the 24
local series — Wilcoxon (Shapiro p ≈ 0, non-Normal), p = 0.0005, median MASE
cost of dropping the regressors = +0.080, rank-biserial r = 0.77, 95% CI
[0.40, 1.00]. The CI is wide (n=24), but excludes zero: the regressor benefit
to the local model is real, not an artifact of one or two series.

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
least valuable group for XGBoost (2.6%, below calendar's 4.1% and promotion's
4.2%) — for LightGBM it's actually **promotion** (1.4%), narrowly below
weather's 1.8% (see the statistical audit below: this asymmetry, and the
model-family interaction generally, holds up under a proper paired test, not
just the raw percentages). The bivariate correlation evidence backs up
weather's low overall value more carefully than the original EDA citation did
(see below), alongside the weak temperature-driver recovery already documented
in the SHAP work.

**Test-selection correction on the weather-correlation citation.** The original
draft of this report cited "temperature correlates only 0.02 with raw sales" as
supporting evidence — that number is the row-level (item × store × day, n=394,560)
**Pearson** r between raw `sales` and raw `temperature`. Raw `sales` is a
right-skewed, zero-inflated count (see RQ1's `zero_rate` column), which is
precisely the profile where Pearson is the wrong tool: re-running the same
row-level comparison with Spearman (`rq2_weather_correlation_audit.py`,
`rq2_weather_correlation.csv`) gives **rho = −0.046, 95% CI [−0.049, −0.043],
p ≈ 0** — the *sign flips* relative to the Pearson figure (+0.019, 95% CI
[0.016, 0.022]). At the two other granularities the EDA notebook actually reports,
Pearson and Spearman agree in both sign and magnitude: aggregated (date×province
sum) sales vs. `temp_anomaly` is +0.172 (Pearson) vs. +0.169 (Spearman, 95% CI
[0.139, 0.197]), and the 30 item-level correlations disagree in sign on only 2 of
30 items, with a bootstrapped median Spearman rho of +0.003 (95% CI [−0.015,
+0.037]) — indistinguishable from zero either way.

**Net effect on the conclusion: none.** Every one of these correlations — Pearson
or Spearman, row-level, aggregated, or item-level — sits within ±0.05 to ±0.17,
which is a negligible effect size by any convention, so "weather is the least
valuable group" still holds. But the *specific number* the report leaned on was
the least robust of the three available (sign-unstable under the
distribution-appropriate test), or a single row-level Pearson r on a skewed count
variable, not the aggregated or item-level figures the EDA notebook actually
headlines. The fix: cite the range across granularities/methods with a direction
sentence, not a single unqualified point estimate — "all weather correlations with
sales sit in the ±0.05 to ±0.17 band regardless of Pearson vs. Spearman, so the
ablation-study accuracy cost (1.8–2.6% WAPE) is the more trustworthy evidence for
'weather adds little' than any bivariate correlation number."

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
