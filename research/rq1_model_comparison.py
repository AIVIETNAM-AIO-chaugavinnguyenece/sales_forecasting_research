"""RQ1 -- Model family comparison: global ML vs local statistical.

Research question: Which of LightGBM, XGBoost and SARIMA achieves the best
accuracy, and under what series conditions does each win?
"""
import numpy as np
import pandas as pd

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    fit_lgbm, fit_xgb, wape, mase, diebold_mariano, N_LOCAL_SERIES,
)

RESULTS_DIR = "results"
MODELS = ["LightGBM", "XGBoost", "SARIMA"]


def fit_global_models(df, cols, fit_mask, val_mask, train_mask):
    X_fit, y_fit = df.loc[fit_mask, cols], df.loc[fit_mask, "sales"]
    X_val, y_val = df.loc[val_mask, cols], df.loc[val_mask, "sales"]
    X_test = df.loc[~train_mask, cols]
    cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]

    lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
    xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)

    return lgbm_model.predict(X_test), xgb_model.predict(X_test)


def fit_local_sarima(df, series_ids):
    panel = df[["date", "store_item", "sales", "is_test"]].copy()
    panel = panel[panel["store_item"].isin(series_ids)].sort_values(["store_item", "date"])
    preds = run_sarimax(panel, series_ids)
    return preds.rename(columns={"pred": "pred_SARIMA"})


def series_conditions(df, train_mask, series_ids):
    tr = df.loc[train_mask]
    tr = tr[tr["store_item"].isin(series_ids)]
    g = tr.groupby("store_item")["sales"]
    return pd.DataFrame({
        "mean_volume": g.mean(),
        "cv": g.std() / g.mean(),
        "zero_rate": g.apply(lambda s: (s == 0).mean()),
    })


def main():
    df = load_features()
    cols = feature_cols(df)
    train_mask, fit_mask, val_mask, cutoff = time_split(df)

    print("=== Fitting global models (LightGBM, XGBoost) ===")
    pred_lgbm, pred_xgb = fit_global_models(df, cols, fit_mask, val_mask, train_mask)

    print("\n=== Fitting local SARIMA (24-series stratified sample) ===")
    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    sarima_preds = fit_local_sarima(df, local_series)

    global_preds = df.loc[~train_mask, ["date", "store_item", "sales"]].copy()
    global_preds["pred_LightGBM"] = pred_lgbm
    global_preds["pred_XGBoost"] = pred_xgb
    comparison = (
        global_preds[global_preds["store_item"].isin(local_series)]
        .rename(columns={"sales": "actual"})
        .merge(sarima_preds, on=["store_item", "date", "actual"], how="inner")
    )
    assert len(comparison) > 0, "comparison frame is empty -- check series id alignment"
    print(f"\ncomparison frame: {len(comparison):,} rows across {comparison['store_item'].nunique()} series")

    train_by_series = df.loc[train_mask].groupby("store_item")["sales"]

    overall_rows = []
    per_series = {m: {} for m in MODELS}
    for m in MODELS:
        col = f"pred_{m}"
        for sid, g in comparison.groupby("store_item"):
            per_series[m][sid] = mase(g["actual"].values, g[col].values, train_by_series.get_group(sid).values)
        overall_rows.append({
            "model": m,
            "MAE": np.mean(np.abs(comparison["actual"] - comparison[col])),
            "RMSE": np.sqrt(np.mean((comparison["actual"] - comparison[col]) ** 2)),
            "WAPE": wape(comparison["actual"].values, comparison[col].values),
            "MASE": np.nanmean(list(per_series[m].values())),
        })
    overall = pd.DataFrame(overall_rows).sort_values("MASE").reset_index(drop=True)
    print("\n=== Overall (on the 24-series subset) ===")
    print(overall.round(4))
    overall.to_csv(f"{RESULTS_DIR}/rq1_overall.csv", index=False)

    cond = series_conditions(df, train_mask, local_series)
    cond["cv_band"] = pd.qcut(cond["cv"], 2, labels=["low volatility", "high volatility"])
    vol = comparison.groupby("store_item")["actual"].mean()
    cond["volume_band"] = pd.qcut(vol.reindex(cond.index), 3, labels=["low volume", "mid volume", "high volume"])

    seg_rows = []
    for seg_col in ["volume_band", "cv_band"]:
        for band, sids in cond.groupby(seg_col, observed=True).groups.items():
            for m in MODELS:
                vals = [per_series[m][sid] for sid in sids if sid in per_series[m]]
                seg_rows.append({"segment_type": seg_col, "segment": band, "model": m,
                                  "MASE": np.nanmean(vals), "n_series": len(vals)})
    seg = pd.DataFrame(seg_rows)
    print("\n=== Segmented MASE ===")
    print(seg.pivot_table(index=["segment_type", "segment"], columns="model", values="MASE", observed=True).round(3))
    seg.to_csv(f"{RESULTS_DIR}/rq1_segmented.csv", index=False)

    errors = {m: (comparison["actual"] - comparison[f"pred_{m}"]).values for m in MODELS}
    dm_rows = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            s, p = diebold_mariano(errors[MODELS[i]], errors[MODELS[j]])
            dm_rows.append({"model_a": MODELS[i], "model_b": MODELS[j], "dm_stat": s,
                             "p_value": p, "equivalent": bool(p > 0.05) if np.isfinite(p) else None})
    dm = pd.DataFrame(dm_rows)
    print("\n=== Diebold-Mariano (pairwise) ===")
    print(dm.round(4))
    dm.to_csv(f"{RESULTS_DIR}/rq1_dm_tests.csv", index=False)

    winners = pd.Series({sid: min(MODELS, key=lambda m: per_series[m].get(sid, np.inf))
                          for sid in local_series if sid in cond.index})
    cond["winner"] = winners
    print("\n=== Winner counts (by per-series MASE) ===")
    print(cond["winner"].value_counts())
    print("\n=== Mean series condition by winner ===")
    print(cond.groupby("winner", observed=True)[["mean_volume", "cv", "zero_rate"]].mean().round(3))
    cond.to_csv(f"{RESULTS_DIR}/rq1_series_conditions.csv")

    print("\nRQ1 evaluation complete.")


if __name__ == "__main__":
    main()
