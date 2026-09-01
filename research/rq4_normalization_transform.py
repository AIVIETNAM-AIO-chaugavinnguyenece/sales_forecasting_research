"""RQ4 -- Normalisation and target transformation.

Research question: How do target transformation and cross-series normalisation
affect forecast accuracy, and does the effect differ systematically between
global tree ensembles and local statistical models?

Pre-registered predictions (fixed before any number below is computed):
  P1  log1p changes global-model WAPE by more than it changes SARIMA's WAPE --
      global models pool heterogeneous-volume series, so a variance-stabilising
      transform should matter more to them than to a model already fit on one
      series' own scale.
  P2  per-series z-score normalisation meaningfully improves the global models
      (putting every series on a comparable scale for the pooled loss).
  P3  falsification: if both families move by a similar amount under log1p, the
      effect is generic feature-engineering noise, not family-specific.
"""
import numpy as np
import pandas as pd

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    fit_lgbm, fit_xgb, wape, mase, SEED, N_LOCAL_SERIES,
)

RESULTS_DIR = "results"


def inverse_log1p(y):
    return np.maximum(np.expm1(y), 0)


def fit_global_variant(df, cols, fit_mask, val_mask, train_mask, transform):
    y_fit_raw = df.loc[fit_mask, "sales"].values
    y_val_raw = df.loc[val_mask, "sales"].values
    X_fit, X_val, X_test = df.loc[fit_mask, cols], df.loc[val_mask, cols], df.loc[~train_mask, cols]
    cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]

    if transform == "raw":
        y_fit, y_val, invert = y_fit_raw, y_val_raw, (lambda y: y)
    elif transform == "log1p":
        y_fit, y_val, invert = np.log1p(y_fit_raw), np.log1p(y_val_raw), inverse_log1p
    else:
        raise ValueError(transform)

    lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
    xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)
    return {"LightGBM": invert(lgbm_model.predict(X_test)), "XGBoost": invert(xgb_model.predict(X_test))}


def fit_global_zscore(df, cols, fit_mask, val_mask, train_mask):
    """Per-series z-score target normalisation, mean/std from training rows only."""
    stats_by_series = df.loc[train_mask].groupby("store_item")["sales"].agg(["mean", "std"])
    stats_by_series["std"] = stats_by_series["std"].replace(0, np.nan).fillna(1)
    series_mean = df["store_item"].map(stats_by_series["mean"])
    series_std = df["store_item"].map(stats_by_series["std"])
    y_norm = (df["sales"] - series_mean) / series_std

    X_fit, X_val, X_test = df.loc[fit_mask, cols], df.loc[val_mask, cols], df.loc[~train_mask, cols]
    y_fit, y_val = y_norm.loc[fit_mask].values, y_norm.loc[val_mask].values
    cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]
    test_mean, test_std = series_mean.loc[~train_mask].values, series_std.loc[~train_mask].values

    lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
    xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)
    return {
        "LightGBM": np.maximum(lgbm_model.predict(X_test) * test_std + test_mean, 0),
        "XGBoost": np.maximum(xgb_model.predict(X_test) * test_std + test_mean, 0),
    }


def fit_local_sarima(panel, series_ids, transform):
    rows, failures = [], 0
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    for sid in series_ids:
        g = panel[panel["store_item"] == sid]
        tr = g[~g["is_test"]].reset_index(drop=True)
        te = g[g["is_test"]].reset_index(drop=True)
        if len(tr) < 60 or len(te) == 0:
            continue
        y_tr = np.log1p(tr["sales"].values) if transform == "log1p" else tr["sales"].values
        try:
            m = SARIMAX(y_tr, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
            fit = m.fit(disp=False, maxiter=100)
            fc = fit.forecast(steps=len(te))
            if transform == "log1p":
                fc = inverse_log1p(fc)
        except Exception as e:
            failures += 1
            print(f"    [fallback] {sid}: {type(e).__name__}: {e}")
            fc = np.repeat(tr["sales"].tail(7).mean(), len(te))
        rows.append(pd.DataFrame({"store_item": sid, "date": te["date"].values,
                                   "actual": te["sales"].values,
                                   "pred": np.maximum(np.asarray(fc, dtype=float), 0)}))
    print(f"  SARIMA[{transform}]: {failures} fallbacks / {len(series_ids)} series")
    return pd.concat(rows, ignore_index=True)


def main():
    df = load_features()
    cols = feature_cols(df)
    train_mask, fit_mask, val_mask, cutoff = time_split(df)
    y_train_full = df.loc[train_mask, "sales"].values
    y_test = df.loc[~train_mask, "sales"].values

    print("=== Global models across target-transform variants ===")
    rows = []
    for transform in ["raw", "log1p"]:
        preds = fit_global_variant(df, cols, fit_mask, val_mask, train_mask, transform)
        for m, p in preds.items():
            rows.append({"model": m, "variant": transform, "MAE": np.mean(np.abs(y_test - p)),
                         "WAPE": wape(y_test, p), "MASE": mase(y_test, p, y_train_full)})
        print(f"  {transform} done")

    zscore_preds = fit_global_zscore(df, cols, fit_mask, val_mask, train_mask)
    for m, p in zscore_preds.items():
        rows.append({"model": m, "variant": "per_series_zscore", "MAE": np.mean(np.abs(y_test - p)),
                     "WAPE": wape(y_test, p), "MASE": mase(y_test, p, y_train_full)})
    print("  per_series_zscore done")

    global_results = pd.DataFrame(rows)
    print("\n" + global_results.round(4).to_string(index=False))
    global_results.to_csv(f"{RESULTS_DIR}/rq4_global_transforms.csv", index=False)
    assert len(global_results) == 6, f"expected 6 rows (2 models x 3 variants), got {len(global_results)}"

    print("\n=== Local SARIMA: raw vs log1p (24-series sample) ===")
    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    panel = df[["date", "store_item", "sales", "is_test"]].copy()
    panel = panel[panel["store_item"].isin(local_series)].sort_values(["store_item", "date"])
    train_by_series = df.loc[train_mask].groupby("store_item")["sales"]

    local_rows = []
    for transform in ["raw", "log1p"]:
        preds = fit_local_sarima(panel, local_series, transform)
        per_series_mase = [mase(g["actual"].values, g["pred"].values, train_by_series.get_group(sid).values)
                            for sid, g in preds.groupby("store_item")]
        local_rows.append({"model": "SARIMA", "variant": transform,
                            "MAE": np.mean(np.abs(preds["actual"] - preds["pred"])),
                            "WAPE": wape(preds["actual"].values, preds["pred"].values),
                            "MASE": np.nanmean(per_series_mase)})
    local_results = pd.DataFrame(local_rows)
    print("\n" + local_results.round(4).to_string(index=False))
    local_results.to_csv(f"{RESULTS_DIR}/rq4_local_transforms.csv", index=False)
    assert len(local_results) == 2, f"expected 2 rows (raw, log1p), got {len(local_results)}"

    print("\n=== Verdict ===")
    g_pivot = global_results.pivot(index="model", columns="variant", values="WAPE")
    global_log_delta_pct = 100 * (g_pivot["raw"] - g_pivot["log1p"]) / g_pivot["raw"]
    global_z_delta_pct = 100 * (g_pivot["raw"] - g_pivot["per_series_zscore"]) / g_pivot["raw"]
    l_pivot = local_results.set_index("variant")["WAPE"]
    local_log_delta_pct = 100 * (l_pivot["raw"] - l_pivot["log1p"]) / l_pivot["raw"]

    print(f"Global WAPE improvement from log1p (%):           {global_log_delta_pct.round(2).to_dict()}")
    print(f"Global WAPE improvement from per-series zscore (%): {global_z_delta_pct.round(2).to_dict()}")
    print(f"Local (SARIMA) WAPE improvement from log1p (%):    {local_log_delta_pct:.2f}")

    P1 = bool((global_log_delta_pct.abs() > abs(local_log_delta_pct)).all())
    P2 = bool((global_z_delta_pct > 1).any())
    P3 = bool((global_log_delta_pct.abs() - abs(local_log_delta_pct)).abs().max() < 1)
    print(f"\nP1 (log1p moves global WAPE more than it moves SARIMA's): {P1}")
    print(f"P2 (z-score meaningfully helps at least one global model): {P2}")
    print(f"P3 (falsification -- effect sizes converge to within 1pt): {P3}")

    print("\nRQ4 evaluation complete.")


if __name__ == "__main__":
    main()
