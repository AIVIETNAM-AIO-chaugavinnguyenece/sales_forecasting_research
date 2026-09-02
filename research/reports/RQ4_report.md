# RQ4 — Normalisation and Target Transformation

**Question:** How do target transformation and cross-series normalisation affect
forecast accuracy, and does the effect differ systematically between global tree
ensembles and local statistical models?

**Codefile:** `research/rq4_normalization_transform.py`, `research/rq4_series_level_test.py`,
`research/rq4_statistical_audit.py`
**Evaluation artifacts:** `research/results/rq4_global_transforms.csv`,
`rq4_local_transforms.csv`, `rq4_global_per_series_mase.csv`,
`rq4_local_per_series_mase.csv`, `rq4_main_effect_tests.csv`,
`rq4_interaction_tests.csv`, `rq4_sarima_zscore_noop_check.csv`

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

*(These P1–P3 verdicts are read directly off the pooled WAPE table — n=1 per
cell, no test. The statistical audit below re-examines them with a proper
paired, per-series test and does not confirm all of them — see "Interpretation
(rewritten)" for the corrected read.)*

## Statistical audit

A principal-level audit (`ml-datasci/auditing-train-test-split` →
`ml-datasci/selecting-statistical-test` → `ml-datasci/checking-test-assumptions`
→ `ml-datasci/reporting-effect-sizes`) re-examined the P1–P3 verdicts above,
which rest on one pooled WAPE number per (model, variant) — n=1, no CI, no test.

### Step 1: consistency check, before trusting any number

Code-audited (no leakage found), then one gap was checked empirically rather
than assumed:
- **log1p is applied and inverted through the same `inverse_log1p` function**
  for LightGBM, XGBoost, and SARIMA — no divergent invert logic between
  families. All three are scored against the same raw `sales` actuals, and
  MASE's naive-denominator always uses raw train sales regardless of variant.
  Train/val/test boundaries are `common.time_split`, already cleared in the
  RQ1 audit. **These numbers are safe to compare across families.**
- **Per-series z-score has no SARIMA arm in the original design.** The report
  already gives a theoretical reason (linear rescaling of a linear Gaussian
  ARMA target doesn't change forecasts once inverted) but never tested it.
  This audit added the arm (`rq4_series_level_test.py`) and confirms it
  empirically: SARIMA raw vs. z-scored per-series MASE, median delta =
  **−0.0000019** (`rq4_sarima_zscore_noop_check.csv`), not significant
  (Wilcoxon p = 0.14). The theory holds — z-score really is a numerical no-op
  for SARIMA — but this means **P2/P3's "does normalisation differ by family"
  claim was only ever testable via log1p**, not via z-score in general; that
  scope limit should have been stated explicitly rather than left implicit.

### Steps 2–3: test selection + assumption check, on real per-series MASE

Same n=1 problem as RQ2/RQ1: refit to recover per-series MASE (360 series,
global; 24 series, local — `rq4_series_level_test.py`) turns "does transform T
help model M" into a paired test and "does T's effect differ by family" into
an interaction (difference-of-differences on the 24 series both families were
scored on — the pairing RQ1's cross-family comparison already established).
Shapiro-Wilk on every paired-difference vector rejected Normality (p < 0.001
in all 10 cases) → Wilcoxon signed-rank throughout, Holm-Bonferroni within
each family of tests (6 main-effect, 4 interaction).

**Main effect (transform vs. raw), per model family** (`rq4_main_effect_tests.csv`):

| model | variant | n | median MASE delta | p (raw) | Holm-significant | rank-biserial r | 95% CI |
|---|---|---|---|---|---|---|---|
| LightGBM | log1p | 360 | −0.014 | <0.0001 | yes | 0.49 | [0.38, 0.59] |
| LightGBM | zscore | 360 | −0.007 | 0.0006 | yes | 0.21 | [0.10, 0.32] |
| XGBoost | log1p | 360 | −0.018 | **0.81** | **no** | 0.01 | [0.00, 0.15] |
| XGBoost | zscore | 360 | +0.036 | <0.0001 | yes | 0.54 | [0.44, 0.63] |
| SARIMA | log1p | 24 | −0.019 | 0.34 | no | 0.23 | [0.01, 0.66] |
| SARIMA | zscore | 24 | −0.0000 | 0.14 | no | 0.35 | [0.03, 0.75] |

Two of these contradict the pooled-WAPE table directly:

- **XGBoost log1p is not a real effect.** The pooled WAPE said "−5.72%,
  worse." The per-series test says p=0.81 — no systematic effect at all.
  The median delta is even slightly negative (typical series marginally
  *improves*), while the mean delta is positive (+0.014) — a classic
  skew signature, confirmed by looking directly at the 360 deltas: **219 of
  360 series (61%) improve** under log1p, only 141 worsen, but the worsening
  tail is longer (max delta +0.31 vs. min −0.23) and concentrated — 4 of the
  5 worst-hit series all share `item_id=7` across different stores. WAPE's
  volume-weighting lets that small, concentrated tail dominate the pooled
  number; the per-series, equally-weighted test shows the typical series is
  unaffected or mildly helped. **"log1p makes XGBoost worse" should be
  retired** in favor of "log1p makes XGBoost worse on a specific, concentrated
  subset of series (notably item 7's stores), with no systematic effect
  across the population."
- **LightGBM z-score is not "flat."** Pooled WAPE showed −0.05% ("no effect").
  The per-series test finds a small but real, Holm-significant improvement
  (r=0.21, CI excludes zero). The pooled number nets out to ~zero because the
  effect is small and MASE-per-series is equally weighted while WAPE is
  volume-weighted — small-and-real is not the same as absent.
- SARIMA's own log1p main effect, at n=24, does **not** clear significance
  (p=0.34) despite the large pooled WAPE swing (+6.69%). This doesn't mean the
  pooled effect is fake — n=24 has limited power, and the point estimate and
  effect-size CI are consistent with a real but noisily-estimated effect — but
  it means the confidence the original draft placed in this number was not
  earned by a matching test.

### Step 4: the interaction — is "differs systematically by family" supported?

Restricted to the 24 series both families were scored on
(`rq4_interaction_tests.csv`), difference-of-differences, Holm-corrected
across the 4 tests:

| variant | family pair | n | median (family Δ − SARIMA Δ) | p (raw) | Holm-significant | rank-biserial r | 95% CI |
|---|---|---|---|---|---|---|---|
| log1p | LightGBM vs SARIMA | 24 | +0.016 | 0.42 | **no** | 0.19 | [0.01, 0.65] |
| log1p | XGBoost vs SARIMA | 24 | +0.036 | 0.14 | **no** | 0.35 | [0.03, 0.75] |
| zscore | LightGBM vs SARIMA | 24 | +0.003 | 0.86 | no | 0.05 | [0.01, 0.53] |
| zscore | XGBoost vs SARIMA | 24 | +0.040 | **0.0096** | **yes** | 0.59 | [0.17, 0.91] |

**Only one of four interactions survives — and it isn't the one the original
report was built on.** Both log1p interactions (the pairing the original
"it's the lag-feature-free local model that benefits from transformation"
narrative rests on) are non-significant, with wide effect-size CIs that
include near-zero. The interaction that *does* survive Holm correction is
XGBoost-vs-SARIMA under z-score (p=0.0096, r=0.59) — which just confirms
XGBoost is scale-sensitive while SARIMA (correctly) isn't, not the
temperature/lag-absorption mechanism the report leans on.

## Interpretation (rewritten — the audit forces this)

**The original "yes, it differs systematically by family, driven by
lag-feature-free SARIMA benefiting most from log1p" answer does not survive
the corrected analysis, and the step that forces the rewrite is step 4 (the
interaction test).** Both log1p family-interaction tests (LightGBM-vs-SARIMA,
XGBoost-vs-SARIMA) came back non-significant (p=0.42, p=0.14) with
effect-size CIs that include near-zero. The mechanistic story — SARIMA has no
lag features so it needs log1p more — is a plausible hypothesis and the point
estimates still point that direction, but at n=24 series the data cannot
distinguish it from noise. The original report treated a single pooled WAPE
swing (+6.69%) as confirmation; a matching interaction test says "consistent
with, but not confirmed by, this sample."

What the corrected analysis **does** support:

- **z-score genuinely hurts XGBoost and genuinely doesn't touch SARIMA** —
  the one interaction (XGBoost vs. SARIMA, z-score) that survives Holm
  correction (p=0.0096, r=0.59). This is a real, family-differentiated
  finding, but it's a story about **XGBoost's optimization sensitivity to
  target scale**, not about lag features vs. no lag features — SARIMA's
  z-score invariance was confirmed to be a numerical no-op (step 1), which
  isn't a finding about "needing" normalisation less, it's the model being
  mathematically indifferent to it.
- **XGBoost's log1p "degradation" isn't systematic** — it's concentrated in a
  minority of series (39%, disproportionately one item's stores) with no
  significant population-level effect (p=0.81). The original claim that
  "XGBoost is the more sensitive of the two to any target transform,
  degrading under both log1p and z-score" is half right: true for z-score
  (large, real, Holm-significant effect, r=0.54), not established for log1p.
- **LightGBM's z-score effect is small but real** (r=0.21, Holm-significant),
  contrary to the original "flat / no effect" read of the pooled WAPE number.

**Revised answer to the RQ:** normalisation choice has real, family-specific
effects, but the *evidence for* a systematic global-vs-local difference is
much narrower than the original draft claimed — it rests on one confirmed
interaction (XGBoost's z-score sensitivity), not on the log1p/lag-absorption
mechanism the original interpretation was built around. That mechanism
remains a reasonable hypothesis for a follow-up study with more local series
(`N_LOCAL_SERIES = None`, all 360), not a supported finding from this one.

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
