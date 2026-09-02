"""RQ1 addendum -- repeated-measures test on the real per-series MASE distribution.

`rq1_model_comparison.py` computes a per-series MASE dict internally
(`per_series[model][series_id]`) but only persists the aggregate and the
per-series *winner*, not the per-series MASE values themselves. The design
behind the RQ1 comparison is: 3 models, each scored on the SAME 24 series
(repeated-measures / paired, not independent samples) -- so the correct test
operates on this 24-row per-series table, not on the pooled 2,208-row
Diebold-Mariano comparison (which treats within-series-autocorrelated,
cross-series-pooled rows as i.i.d. -- a pseudo-replication risk the DM test
in `rq1_statistical_trio.py` does not address). This script re-fits the same
models via `common.py` (identical split, identical local-series sample, same
seed) purely to recover and persist that per-series table, then:

  1. selecting-statistical-test: 3 groups, repeated-measures, continuous ->
     repeated-measures ANOVA if Normal, Friedman (Demsar 2006) if not.
  2. checking-test-assumptions: Shapiro-Wilk on the three pairwise per-series
     MASE differences (the actual gating check for step 1, not an assumption).
  3. Runs the test the assumption check selects, plus Nemenyi/Wilcoxon+Holm
     post-hoc if the omnibus rejects.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    fit_lgbm, fit_xgb, mase, N_LOCAL_SERIES,
)

RESULTS_DIR = "results"
MODELS = ["LightGBM", "XGBoost", "SARIMA"]


def main():
    df = load_features()
    cols = feature_cols(df)
    train_mask, fit_mask, val_mask, cutoff = time_split(df)

    X_fit, y_fit = df.loc[fit_mask, cols], df.loc[fit_mask, "sales"]
    X_val, y_val = df.loc[val_mask, cols], df.loc[val_mask, "sales"]
    X_test = df.loc[~train_mask, cols]
    cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]

    print("=== Fitting global models ===")
    lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
    xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)
    pred_lgbm, pred_xgb = lgbm_model.predict(X_test), xgb_model.predict(X_test)

    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    print("\n=== Fitting local SARIMA ===")
    panel = df[["date", "store_item", "sales", "is_test"]].copy()
    panel = panel[panel["store_item"].isin(local_series)].sort_values(["store_item", "date"])
    sarima_preds = run_sarimax(panel, local_series).rename(columns={"pred": "pred_SARIMA"})

    global_preds = df.loc[~train_mask, ["date", "store_item", "sales"]].copy()
    global_preds["pred_LightGBM"] = pred_lgbm
    global_preds["pred_XGBoost"] = pred_xgb
    comparison = (
        global_preds[global_preds["store_item"].isin(local_series)]
        .rename(columns={"sales": "actual"})
        .merge(sarima_preds, on=["store_item", "date", "actual"], how="inner")
    )

    train_by_series = df.loc[train_mask].groupby("store_item")["sales"]
    rows = []
    for sid, g in comparison.groupby("store_item"):
        row = {"store_item": sid}
        for m in MODELS:
            row[m] = mase(g["actual"].values, g[f"pred_{m}"].values, train_by_series.get_group(sid).values)
        rows.append(row)
    per_series = pd.DataFrame(rows).set_index("store_item")
    per_series.to_csv(f"{RESULTS_DIR}/rq1_per_series_mase.csv")
    print(f"\nper-series MASE table ({len(per_series)} series):")
    print(per_series.round(4).to_string())

    # --- Step 2/3: gating assumption check, then the test it selects ---
    pairs = [("LightGBM", "XGBoost"), ("LightGBM", "SARIMA"), ("XGBoost", "SARIMA")]
    print("\n=== Step 3: Shapiro-Wilk on pairwise per-series MASE differences ===")
    shapiro_rows = []
    for a, b in pairs:
        d = (per_series[a] - per_series[b]).dropna()
        w, p = stats.shapiro(d)
        shapiro_rows.append({"pair": f"{a}-{b}", "n": len(d), "shapiro_W": w, "shapiro_p": p,
                              "normal_at_05": p > 0.05})
        print(f"  {a} - {b}: n={len(d)}, Shapiro W={w:.4f}, p={p:.4f}, "
              f"{'Normal' if p > 0.05 else 'NOT Normal'}")
    shapiro_df = pd.DataFrame(shapiro_rows)
    shapiro_df.to_csv(f"{RESULTS_DIR}/rq1_shapiro_differences.csv", index=False)

    all_normal = bool(shapiro_df["normal_at_05"].all())
    print(f"\nAll three pairwise differences Normal at alpha=0.05: {all_normal}")

    print("\n=== Omnibus test ===")
    if all_normal:
        # f_oneway would be wrong here -- it assumes independent groups and
        # discards the pairing (same 24 series measured under all 3 models).
        # AnovaRM is the actual repeated-measures ANOVA on this long-format table.
        from statsmodels.stats.anova import AnovaRM
        long = per_series.reset_index().melt(id_vars="store_item", value_vars=MODELS,
                                              var_name="model", value_name="mase")
        res = AnovaRM(long, depvar="mase", subject="store_item", within=["model"]).fit()
        f_stat = res.anova_table["F Value"].iloc[0]
        p_omni = res.anova_table["Pr > F"].iloc[0]
        print(f"Repeated-measures ANOVA: F={f_stat:.4f}, p={p_omni:.4g}")
        omni = {"test": "repeated-measures ANOVA", "stat": f_stat, "p": p_omni}
    else:
        chi2, p_omni = stats.friedmanchisquare(*[per_series[m].values for m in MODELS])
        print(f"Friedman chi-square: {chi2:.4f}, p={p_omni:.4g}")
        omni = {"test": "Friedman", "stat": chi2, "p": p_omni}
    pd.DataFrame([omni]).to_csv(f"{RESULTS_DIR}/rq1_omnibus_test.csv", index=False)

    print("\n=== Post-hoc pairwise (Wilcoxon signed-rank, Holm-Bonferroni corrected) ===")
    posthoc_rows = []
    pvals = []
    for a, b in pairs:
        d = per_series[a] - per_series[b]
        stat, p = stats.wilcoxon(per_series[a], per_series[b])
        pvals.append(p)
        posthoc_rows.append({"pair": f"{a} vs {b}", "wilcoxon_stat": stat, "p_raw": p,
                              "median_delta": d.median()})
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        alpha_step = 0.05 / (m - rank)
        if pvals[idx] < alpha_step:
            reject[idx] = True
        else:
            break
    for i, r in enumerate(posthoc_rows):
        r["holm_reject"] = reject[i]
    posthoc = pd.DataFrame(posthoc_rows)
    print(posthoc.round(4).to_string(index=False))
    posthoc.to_csv(f"{RESULTS_DIR}/rq1_posthoc_wilcoxon.csv", index=False)

    print("\nRQ1 series-level test complete.")


if __name__ == "__main__":
    main()
