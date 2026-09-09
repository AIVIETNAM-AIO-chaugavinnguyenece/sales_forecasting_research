# Agreement Is Not Correctness

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-9cf)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red)
![SHAP](https://img.shields.io/badge/SHAP-0.52.0-orange)
![statsmodels](https://img.shields.io/badge/statsmodels-0.15.0-informational)
![License](https://img.shields.io/badge/License-MIT-green)

Model families, external regressors, and explanation faithfulness in retail sales forecasting —
a controlled study on synthetic Australian retail data with **held-out ground-truth drivers**.

This repository contains the data generator, preprocessing pipeline, four experiment scripts,
per-experiment result artefacts, and the analysis notebooks for the **AI VIET NAM — AI Conquer 2026**
research project. The complete methodology, results and discussion are presented in the technical report.

---

## Topic

Explainable sales forecasting: comparing global machine-learning and local statistical forecasters,
quantifying the value of exogenous regressors, and **adjudicating between competing attribution
methods** against demand drivers that are known exactly and withheld from every model.

## Brief Topic Description

Retail forecasting moved to global gradient-boosted models, which forecast better than the
statistical methods they replaced and explain nothing. The field's answer is post-hoc attribution —
in practice, SHAP.

There is a validation gap underneath. **SHAP is faithful to the *model*, not necessarily correct
about the *world*.** If a model routes a genuine causal driver through a correlated proxy feature,
SHAP reports the proxy — accurately, and misleadingly. Practitioners fill the gap with a heuristic:
if two explainers agree, the explanation is trusted. That heuristic has a structural blind spot —
**two methods can agree and both be wrong** — and on real data nothing distinguishes that case from
two methods agreeing because both are right. A real demand series does not come with its drivers labelled.

This project closes the gap by construction. The panel is synthetic and parameterised, so every
item's true temperature sensitivity and price elasticity is known exactly, quarantined from every
model, and used only to score the explanations at the end.

> **Headline finding:** SHAP importances from LightGBM and XGBoost correlate at **ρ = 0.963** — yet
> on the true temperature drivers **both fail** (ρ = 0.365 and 0.408, neither surviving Holm
> correction; recall@5 = 0/5) where a **SARIMAX coefficient succeeds** (ρ = 0.809, recall@5 = 2/5).
> On price the ordering inverts. **Mutual agreement between explainers is not evidence of correctness.**

## Research Questions

> **RQ1 — Model family comparison.** Which of LightGBM, XGBoost and SARIMA achieves the best forecast
> accuracy, and under what series conditions does each perform best?
>
> **RQ2 — Do the external factors earn their place?** How much forecast accuracy do weather, calendar
> and promotion regressors contribute, and does the contribution differ by model family?
>
> **RQ3 — Attribution agreement and recovery of known drivers.** Do SHAP attributions from LightGBM
> and XGBoost agree with each other, with permutation importance, and with SARIMAX coefficients — and
> do they recover the item-level drivers the generator actually used?
>
> **RQ4 — Normalisation and target transformation.** How do target transformation and cross-series
> normalisation affect forecast accuracy, and does the effect differ systematically between global
> tree ensembles and local statistical models?

RQ3 is the core question and is deliberately two questions in one: mutual agreement is what the
existing literature can measure; recovery of held-out parameters is what it cannot.

---

## Experimental Design

| Component | Role |
|---|---|
| **Experimental variables** | Model family (`LightGBM`, `XGBoost`, `SARIMA`) · Regressor group (`weather`, `calendar`, `promotion`) · Attribution method (`SHAP`, `permutation`, `SARIMAX coefficients`) · Target transformation (`raw`, `log1p`, `per-series z`) |
| **Control variables** | One frozen 62-feature matrix · one date-based split · `SEED = 2025` · fixed SARIMA order $(1,1,1)\times(1,0,1)_7$ · fixed hyperparameters · the same 24-series local sample in every experiment |
| **Accuracy measurement** | MAE · RMSE · WAPE · **MASE** (primary, computed per series then averaged) |
| **Significance testing** | Diebold–Mariano for forecast accuracy · Shapiro–Wilk → **Friedman** → Wilcoxon post-hoc at the series level (the correct unit of replication) · rank-biserial effect sizes with bootstrap CIs |
| **Multiple comparisons** | **Holm–Bonferroni** within each RQ family (`common.holm_adjust`) |
| **Attribution agreement** | Spearman $\rho$ and Kendall $\tau$ between global importance rankings over all 62 features |
| **Attribution correctness** | Spearman $\rho$ between per-item attribution mass and the **quarantined** `temp_effect` / `elasticity`, plus population-independent **recall@k** |
| **Falsification arm** | Pre-registered lag-absorption hypothesis with an item-level bootstrap (`notebooks/06_proof_lag_absorption.ipynb`) |

**Five hypotheses were pre-registered before any result was computed.** Two were falsified, one is
undetermined, and a sixth (lag absorption) was falsified outright by our own experiment. All are
reported as such rather than reinterpreted.

---

## Data

| | |
|---|---|
| **Source** | Synthetic — generated by [`data/data_generator.py`](data/data_generator.py), seeded and byte-reproducible |
| **Geography** | Sydney (NSW) · Melbourne (VIC) · Brisbane (QLD) · Adelaide (SA) |
| **Scope** | 12 stores (3 per city) · 30 items across 5 categories · 1,096 days (2023-01-01 → 2025-12-31) |
| **Sample size** | **394,560 observations** across **360 store–item series** |
| **Imperfections** | 1.00% injected missing values · 0.20% injected outliers |
| **Tables** | `sales_data.csv` (394,560) · `weather_data.csv` (4,384) · `holiday_data.csv` (4,384) · `promotion_data.csv` (7,177, one row per campaign) |

Demand is generated multiplicatively and drawn from a Poisson distribution:

```
λ = base × store_size × seasonal(peak_month) × weekday
      × (1 + temp_effect × temperature_anomaly)
      × holiday × (1 − discount)^elasticity × promo_mechanic × growth^years
```

### Three generator decisions that avoid artefacts

| Decision | Why |
|---|---|
| **Southern Hemisphere seasonality** | Ice cream peaks in January (23.65 vs 13.36 in July); coffee inverts (7.96 vs 14.13). A Northern-Hemisphere specification reads backwards. |
| **Temperature against a local norm** | 25 °C is mild in Brisbane, hot in Melbourne. Raw temperature confounds the weather response with a city fixed effect. |
| **Promotions as contiguous 5–14 day blocks** | Independent daily coin flips produce a flickering flag no retailer operates, and make the promotion flag trivially decodable. |

### The state-holiday natural experiment

Australian public holidays vary by state — Labour Day is March in Victoria, May in Queensland,
October in NSW and SA; Melbourne Cup is Victoria-only. **15 dates are public holidays in some states
but not others**, with weather, weekday and season held constant. This arises from faithful
implementation of the statutory rules rather than by design, which is what makes it usable as an
identification strategy.

Full schema and join keys: [`data/README.md`](data/README.md).

---

## The Ground-Truth Quarantine

Each of the 30 items carries two parameters defining its true response:

- **`temp_effect`** — from −0.045 (firelighters, sells more when cold) to +0.060 (sunscreen).
  **11 items are exactly zero**, deliberately: a faithful method must rank them low.
- **`elasticity`** — from −0.40 (full-cream milk, inelastic) to −1.50 (potato chips).

Enforcement is auditable at four points:

| # | Enforcement | Meaning |
|---|---|---|
| 1 | **Not emitted** | No column in any of the four CSVs derives from them; their effect appears only through realised sales |
| 2 | **Not engineered** | No feature is a function of them; item identity enters as a categorical code carrying no parameter information |
| 3 | **Not used in model selection** | Hyperparameters, early-stopping iterations and the SARIMA order were chosen without reference to them |
| 4 | **Loaded separately** | Read from `common.ITEM_TRUTH` at evaluation time only, after every model is fitted and every attribution computed |

---

## Repository Structure

```text
sales_forecasting_research/
├── data/
│   ├── data_generator.py                 # Synthetic generator — the source of ground truth
│   ├── README.md                         # Dataset schema, join keys, driver definitions
│   ├── sales_data.csv                    # 394,560 rows · store × item × day
│   ├── weather_data.csv                  # 4,384 rows · city × day
│   ├── holiday_data.csv                  # 4,384 rows · state × day
│   ├── promotion_data.csv                # 7,177 rows · one row per campaign
│   ├── *_preprocessed.csv                # Output of notebook 01
│   └── feature_engineered_data_69_features.parquet   # Model matrix (62 features + keys)
├── notebooks/                            # Exploratory pipeline, run in order
│   ├── 01_preprocessing.ipynb            # Joins, causal fills, leak removal
│   ├── 02_EDA_updated.ipynb              # Series characteristics, seasonality, driver EDA
│   ├── 03_feature_engineering_updated.ipynb   # 62 features, train-only quantile binning
│   ├── 04_modelling_update.ipynb         # Five models incl. Prophet and ARIMA
│   ├── 05_explain_model.ipynb            # SHAP, permutation importance, SARIMAX coefficients
│   ├── 06_proof_lag_absorption.ipynb     # Pre-registered hypothesis — falsified
│   └── run_lag_absorption.py             # Memory-lean version of notebook 06
├── research/                             # Reproducible experiment layer
│   ├── common.py                         # Shared split, metrics, model configs, ITEM_TRUTH
│   ├── rq1_model_comparison.py           # RQ1
│   ├── rq2_external_factors.py           # RQ2
│   ├── rq3_attribution_agreement.py      # RQ3
│   ├── rq4_normalization_transform.py    # RQ4
│   ├── rq1_rq2_additional.ipynb          # Series-level re-tests (Friedman, Wilcoxon)
│   ├── reports/RQ{1..4}_report.md        # Per-RQ method notes, caveats, known issues
│   ├── results/*.csv                     # 11 result artefacts
│   └── requirements.txt                  # Pinned dependencies
├── models/
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.json               # Feature list, split dates, config, results
├── utils/                                # Plotting and shared helpers
├── figures/                              # EDA and result figures
└── README.md
```

**`research/` is the reproducible core; `notebooks/` is the exploratory record.** Every number cited
in the technical report comes from `research/results/`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-org>/sales_forecasting_research.git
cd sales_forecasting_research
```

Create a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r research/requirements.txt
```

---

## Running the Project

Run the following from the repository root.

### 1. Regenerate the dataset (optional — CSVs are committed)

```bash
cd data && python data_generator.py
```

Writes the four raw tables. Seeded at 2025 and byte-reproducible.

### 2. Run the four experiments

```bash
cd research
python rq1_model_comparison.py          # ~3 min  → results/rq1_*.csv
python rq2_external_factors.py          # ~8 min  → results/rq2_*.csv
python rq3_attribution_agreement.py     # ~6 min  → results/rq3_*.csv
python rq4_normalization_transform.py   # ~5 min  → results/rq4_*.csv
```

Each script prints its tables to stdout and writes tidy CSVs to `research/results/`. All four read
the same feature matrix, use the same split and the same 24-series local sample, so their outputs
are directly comparable.

> **Memory note.** RQ3 computes interventional TreeSHAP; cost scales with
> `N_EXPLAIN × N_BACKGROUND`. Defaults suit roughly 4 GB of free RAM — lower both if you hit pressure.

### 3. Run the series-level re-tests

```bash
jupyter lab research/rq1_rq2_additional.ipynb
```

Shapiro–Wilk normality → Friedman omnibus → Wilcoxon post-hoc with Holm correction, at the correct
unit of replication (24 series rather than 2,208 autocorrelated rows).

### 4. Run the falsification arm

```bash
cd notebooks && python run_lag_absorption.py
```

The pre-registered lag-absorption experiment, with an item-level bootstrap. Memory-lean; the
notebook version is `06_proof_lag_absorption.ipynb`.

### 5. Work through the notebooks

```bash
jupyter lab notebooks/
```

---

## Output Data

Each result file is tidy: one row per experimental condition.

| File | Rows | Fields |
|---|---|---|
| `rq1_overall.csv` | 3 | `model`, `MAE`, `RMSE`, `WAPE`, `MASE` |
| `rq1_dm_tests.csv` | 3 | `model_a`, `model_b`, `dm_stat`, `p_value`, `equivalent` |
| `rq1_segmented.csv` | 15 | `segment_type`, `segment`, `model`, `MASE`, `n_series` |
| `rq1_series_conditions.csv` | 24 | `store_item`, `mean_volume`, `cv`, `zero_rate`, `cv_band`, `volume_band`, `winner` |
| `rq2_global_ablation.csv` | 8 | `model`, `variant`, `n_features`, `MAE`, `WAPE`, `MASE`, `wape_cost_pct` |
| `rq2_local_regressor_value.csv` | 2 | `model`, `MAE`, `WAPE`, `MASE` |
| `rq3_attribution_agreement.csv` | 3 | `comparison`, `rho`, `p_value` |
| `rq3_recovery.csv` | 6 | `driver`, `method`, `rho`, `p_value` |
| `rq3_sarimax_coefficients.csv` | 24 | `store_item`, `item_id`, + one column per exogenous regressor |
| `rq4_global_transforms.csv` | 6 | `model`, `variant`, `MAE`, `WAPE`, `MASE` |
| `rq4_local_transforms.csv` | 2 | `model`, `variant`, `MAE`, `WAPE`, `MASE` |

Results are stored at the individual condition level to support aggregation, effect-size
computation and paired statistical testing.

---

## Results

### RQ1 — Global ML wins under every series condition

| Model | Type | MAE | RMSE | WAPE | **MASE** |
|---|---|---|---|---|---|
| **LightGBM** | Global | **5.276** | **7.933** | **23.47%** | **0.724** |
| XGBoost | Global | 5.693 | 8.403 | 25.32% | 0.790 |
| SARIMA | Local | 9.050 | 12.777 | 40.26% | 1.241 |

At the correct unit of replication: Friedman **χ² = 46.08, p = 9.8 × 10⁻¹¹**; Wilcoxon post-hoc with
Holm confirms **LightGBM has the lower MASE in 24 of 24 series** (rank-biserial r = 1.00). The
conclusion survives a correction that could have overturned it.

**H1 falsified on volume** (SARIMA's gap is smallest at *low* volume, not high) and **undetermined on
volatility** (Kruskal–Wallis p = 0.073, so we decline to claim it). Detectable instead: SARIMA
narrows on sparser series (ρ = −0.45, p = 0.028).

### RQ2 — Regressors earn their place, but the ranking is model-specific

| Removed group | LightGBM (WAPE cost) | XGBoost (WAPE cost) |
|---|---|---|
| Calendar | **+10.51%** | +4.06% |
| Weather | +1.79% | +2.59% |
| Promotion | +1.41% | **+4.17%** |

The same feature group is worth **2.6× more** to one gradient-boosted ensemble than another on
identical data, and the ordering itself flips. **H2 partially supported.** For the local model, six
exogenous regressors are worth **−14.2% MASE** (1.241 → 1.065).

### RQ3 — Agreement without correctness

| Driver | Method | Spearman ρ | Survives Holm | Recall@5 |
|---|---|---|---|---|
| Temperature | **SARIMAX coefficient** | **0.809** | ✔ | **2 / 5** |
| Temperature | SHAP (XGBoost) | 0.408 | ✘ | 0 / 5 |
| Temperature | SHAP (LightGBM) | 0.365 | ✘ | 0 / 5 |
| Price | **SHAP (LightGBM)** | **0.566** | ✔ | 1 / 5 |
| Price | **SHAP (XGBoost)** | **0.527** | ✔ | 2 / 5 |
| Price | SARIMAX coefficient | 0.333 | ✘ | 2 / 5 |

The competencies are **inverted**: neither method dominates, and recovery difficulty is a property of
the **driver–method pair**, not of the driver. **H4 falsified.** A corollary worth stating: the
best-forecasting model is not the best-explained model.

### RQ4 — Transformation effects are model-specific

| Model | raw | log1p | per-series z | Δ log1p |
|---|---|---|---|---|
| LightGBM | 0.752 | **0.745** | 0.752 | −0.98% |
| XGBoost | **0.804** | 0.850 | 0.875 | **+5.72%** |
| SARIMA | 1.241 | **1.156** | — | **−6.90%** |

**H5 supported** — SARIMA most affected, LightGBM least. But XGBoost sits closer to SARIMA than to
LightGBM, so "the tree ensembles" is not one behavioural category. Note the consequence: had
per-series standardisation been adopted as a default, XGBoost would have scored 0.875 rather than
0.804 — **the preprocessing choice would have partly determined the model comparison.**

---

## Team & Contributions

| Member | Role | Responsibilities |
|---|---|---|
| **Ngo Lam Thy** | ML Engineer (local models & XAI) | Local model implementation (SARIMA, SARIMAX); statistical analysis protocol and multiple-comparison correction; RQ3 attribution agreement and driver recovery; RQ4 transformation experiments. |
| **Ngo Huy Hoang** | ML Engineer (global models) | Feature engineering; global model implementation (LightGBM, XGBoost); RQ1 model comparison and significance testing; RQ2 leave-one-group-out ablation. |
| **Vo Quang Ha** | Data & Analysis Engineer | Exploratory data analysis; data preprocessing; groundtruth quarantine protocol; video and presentation. |
| **Nguyen Tran Minh Chau** | Technical Lead & Project Manager | Repository setup; data generation and synthetic panel
design; original codebase creation and code review; master pipeline orchestration; technical report review and finalisation.|

All members reviewed the final manuscript.

---

## Reproducibility Note

The evaluation workflow includes a strictly date-based split with a validation window carved from the
end of training (never the test set), Holm–Bonferroni correction applied within each research-question
family, series-level significance testing at the correct unit of replication, bootstrap confidence
intervals over items for the falsification arm, an auditable four-point ground-truth quarantine, and
pinned dependency versions.

See [`research/reports/RQ{1..4}_report.md`](research/reports/) for per-experiment methodology,
caveats and known issues.

### Known issues

We would rather state these than have them found.

- **XGBoost is not bit-reproducible.** With `tree_method="hist"` and `n_jobs=-1`, histogram
  construction varies between runs even with a fixed `random_state`. A re-execution returned 0.7335
  MASE rather than 0.7903, **flipping the LightGBM/XGBoost Diebold–Mariano verdict to equivalent
  (p = 0.213)**. LightGBM and SARIMA reproduced exactly. Re-run with `n_jobs=1` for a deterministic answer.
- **A silent failure we caught.** The first SARIMAX run raised on all 24 series — a `statsmodels`
  date-index quirk when `exog` is sliced from a multi-series panel — and a broad `except` substituted
  a naive 7-day mean each time, producing a plausible-looking false result. Fixed; the handler now
  prints every fallback it takes.
- **The promotion group is one signal counted twice.** `is_promotion` and `discount_pct` correlate at
  r = 0.90, so the margin by which promotion outranks calendar for XGBoost sits inside that ambiguity.
- **Local results use a 24-series sample.** `N_LOCAL_SERIES` in `common.py`; setting it to `None`
  lifts all four RQs to the full 360-series population at the cost of compute alone.
- **Single seed.** Only the lag-absorption arm reports seed variance, and it showed price recovery
  varying with SD 0.227 — larger than several reported differences.
- **Synthetic data is an upper bound.** Effect sizes are the ones we assigned, and the generator is
  multiplicative while SHAP decomposes additively. Real demand data will be worse.
- **Attributions are associational.** High SHAP attribution on price is a statement about the model's
  function, not about elasticity.

---

## Future Work

1. **Re-run on the full 360-series population** — one line, lifts all four RQs at once.
2. **Deterministic re-execution** with `n_jobs=1` — settles the LightGBM/XGBoost equivalence question.
3. **Seed variance across the full protocol** — report bands, not point estimates.
4. Perturbation sensitivity: interventional vs tree-path-dependent SHAP.
5. **Vary driver functional form while holding magnitude fixed** — tests our most interesting untested
   conjecture, that each method recovers best the drivers whose functional form matches its model's
   inductive bias.
6. Replicate on M5 / Corporación Favorita with partial ground truth.
7. Add N-BEATS, TFT and LIME — a fourth attribution mechanism.

---

## Resources

- **Technical report** — complete methodology, results and discussion (28 references)
- **Environment** — Python 3.12 · LightGBM 4.7.0 · XGBoost 3.4.1 · SHAP 0.52.0 · statsmodels 0.15.0 ·
  scikit-learn 1.8.0 · pandas · numpy · scipy · pyarrow. Seed 2025 throughout.

### Key references

Lundberg & Lee (2017) SHAP · Lundberg et al. (2020) TreeSHAP · Aas et al. (2021) feature dependence ·
Kumar et al. (2020) problems with Shapley importance · Krishna et al. (2024) the disagreement problem ·
Chen et al. (2020) true to the model or the data · Makridakis et al. (2022) M5 ·
Montero-Manso & Hyndman (2021) locality and globality · Hyndman & Koehler (2006) MASE ·
Diebold & Mariano (1995) · Holm (1979) · Friedman (1937)

## Citation

```bibtex
@techreport{agreement2026,
  title       = {Agreement Is Not Correctness: Model Families, External Regressors,
                 and Explanation Faithfulness in Retail Sales Forecasting},
  author      = {Ngo, Huy Hoang and Vo, Quang Ha and Ngo, Lam Thy and Nguyen, Tran Minh Chau},
  year        = {2026},
  institution = {AI VIET NAM --- AI Conquer 2026},
  type        = {Technical Report}
}
```

## License

MIT. The synthetic dataset and generator are released under the same terms — if you build a
faithfulness benchmark on top of it, the ground truth is exact and the parameters are in
[`data/data_generator.py`](data/data_generator.py).
