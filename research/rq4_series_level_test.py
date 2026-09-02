"""RQ4 addendum -- per-series MASE for every (model, transform) combination,
plus an empirical check of whether per-series z-score really is a no-op for
SARIMA (the original script asserts this on theoretical grounds -- linear
rescaling of a linear Gaussian ARMA target doesn't change forecasts once
inverted -- but never actually ran the arm to confirm it).

Consistency check (this RQ's step 1) done by code inspection before any refit:
  - log1p is applied and inverted through the SAME `inverse_log1p` function
    for LightGBM, XGBoost, and SARIMA (`rq4_normalization_transform.py`
    imports it once, uses it in both `fit_global_variant` and
    `fit_local_sarima`) -- no risk of divergent invert logic between families.
  - All three families are scored against the same raw `sales` actuals, and
    MASE's naive-denominator always uses raw (untransformed) train sales --
    consistent regardless of variant.
  - Per-series z-score train mean/std are computed from TRAIN rows only and
    mapped onto fit/val/test alike -- no leakage into val or test.
  - The one real gap: z-score has NO SARIMA arm in the original design. This
    script adds one (a per-series mean/std normalize-fit-invert cycle on a
    single series is a well-defined operation even though theory predicts it
    changes nothing) so the "does normalisation differ by family" question
    can be answered for z-score too, not just log1p.

Per-series MASE (not one pooled number per variant) turns "does the
transform help this family" into a paired test and "does it differ between
families" into an answerable interaction question -- same design fix already
applied in `rq1_series_level_test.py` / `rq2_series_level_test.py`.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    fit_lgbm, fit_xgb, mase, N_LOCAL_SERIES,
)

RESULTS_DIR = "results"
GLOBAL_VARIANTS = ["raw", "log1p", "zscore"]
LOCAL_VARIANTS = ["raw", "log1p", "zscore"]


def inverse_log1p(y):
    return np.maximum(np.expm1(y), 0)


def per_series_mase_from_preds(actual, pred, store_item, train_by_series):
    d = pd.DataFrame({"store_item": store_item, "actual": actual, "pred": pred})
    rows = []
    for sid, g in d.groupby("store_item"):
        rows.append({"store_item": sid,
                     "mase": mase(g["actual"].values, g["pred"].values,
                                  train_by_series.get_group(sid).values)})
    return pd.DataFrame(rows).set_index("store_item")["mase"]


def fit_global_one(df, cols, fit_mask, val_mask, train_mask, transform):
    X_fit, X_val, X_test = df.loc[fit_mask, cols], df.loc[val_mask, cols], df.loc[~train_mask, cols]
    cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]

    if transform == "raw":
        y_fit = df.loc[fit_mask, "sales"].values
        y_val = df.loc[val_mask, "sales"].values
        invert = lambda p, idx: p
    elif transform == "log1p":
        y_fit = np.log1p(df.loc[fit_mask, "sales"].values)
        y_val = np.log1p(df.loc[val_mask, "sales"].values)
        invert = lambda p, idx: inverse_log1p(p)
    elif transform == "zscore":
        stats_by_series = df.loc[train_mask].groupby("store_item")["sales"].agg(["mean", "std"])
        stats_by_series["std"] = stats_by_series["std"].replace(0, np.nan).fillna(1)
        series_mean = df["store_item"].map(stats_by_series["mean"])
        series_std = df["store_item"].map(stats_by_series["std"])
        y_norm = (df["sales"] - series_mean) / series_std
        y_fit = y_norm.loc[fit_mask].values
        y_val = y_norm.loc[val_mask].values
        test_mean, test_std = series_mean.loc[~train_mask].values, series_std.loc[~train_mask].values
        invert = lambda p, idx: np.maximum(p * test_std + test_mean, 0)
    else:
        raise ValueError(transform)

    lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
    xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)
    return invert(lgbm_model.predict(X_test), None), invert(xgb_model.predict(X_test), None)


def fit_local_one(panel, series_ids, transform):
    rows, failures = [], 0
    for sid in series_ids:
        g = panel[panel["store_item"] == sid]
        tr = g[~g["is_test"]].reset_index(drop=True)
        te = g[g["is_test"]].reset_index(drop=True)
        if len(tr) < 60 or len(te) == 0:
            continue
        raw_tr = tr["sales"].values
        if transform == "raw":
            y_tr, invert = raw_tr, (lambda fc: fc)
        elif transform == "log1p":
            y_tr, invert = np.log1p(raw_tr), inverse_log1p
        elif transform == "zscore":
            m, s = raw_tr.mean(), raw_tr.std(ddof=1)
            s = s if (np.isfinite(s) and s > 0) else 1.0
            y_tr = (raw_tr - m) / s
            invert = lambda fc, m=m, s=s: np.maximum(fc * s + m, 0)
        else:
            raise ValueError(transform)
        try:
            model = SARIMAX(y_tr, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                             enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=100)
            fc = fit.forecast(steps=len(te))
            fc = invert(fc)
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
    store_item_test = df.loc[~train_mask, "store_item"].values
    y_test_raw = df.loc[~train_mask, "sales"].values
    train_by_series = df.loc[train_mask].groupby("store_item")["sales"]

    print("=== Global: per-series MASE for LightGBM/XGBoost x {raw, log1p, zscore} ===")
    wide = {}
    for variant in GLOBAL_VARIANTS:
        pred_lgbm, pred_xgb = fit_global_one(df, cols, fit_mask, val_mask, train_mask, variant)
        wide[f"LightGBM_{variant}"] = per_series_mase_from_preds(y_test_raw, pred_lgbm, store_item_test, train_by_series)
        wide[f"XGBoost_{variant}"] = per_series_mase_from_preds(y_test_raw, pred_xgb, store_item_test, train_by_series)
        print(f"  {variant} done")
    global_per_series = pd.DataFrame(wide)
    global_per_series.to_csv(f"{RESULTS_DIR}/rq4_global_per_series_mase.csv")
    print(f"saved {global_per_series.shape} to rq4_global_per_series_mase.csv")

    print("\n=== Local: per-series MASE for SARIMA x {raw, log1p, zscore} (24-series sample) ===")
    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    panel = df[["date", "store_item", "sales", "is_test"]].copy()
    panel = panel[panel["store_item"].isin(local_series)].sort_values(["store_item", "date"])
    local_train_by_series = df.loc[train_mask].groupby("store_item")["sales"]

    local_wide = {}
    for variant in LOCAL_VARIANTS:
        preds = fit_local_one(panel, local_series, variant)
        local_wide[variant] = pd.Series({
            sid: mase(g["actual"].values, g["pred"].values, local_train_by_series.get_group(sid).values)
            for sid, g in preds.groupby("store_item")
        })
    local_per_series = pd.DataFrame(local_wide)
    local_per_series.index.name = "store_item"
    local_per_series.to_csv(f"{RESULTS_DIR}/rq4_local_per_series_mase.csv")
    print(f"saved {local_per_series.shape} to rq4_local_per_series_mase.csv")

    print("\nRQ4 series-level refit complete.")


if __name__ == "__main__":
    main()
