"""RQ2 addendum -- recover per-series accuracy deltas for a real statistical test.

`rq2_external_factors.py` fits each (model, variant) pair exactly once and
reports a single pooled WAPE/MASE over the full 33,120-row test set. That is
n=1 per comparison -- there is no distribution to run a hypothesis test on,
and no way to ask whether the weather/calendar/promotion contribution differs
by model family without collapsing 360 series into one number per side.

This script re-fits the same 8 (model x variant) combinations from
`rq2_external_factors.py` and the 2 local (SARIMA vs SARIMAX) fits from
`rq1_model_comparison.py`'s local-series sample, but persists per-series MASE
so `rq2_statistical_audit.py` can treat the series as the paired unit of
replication -- same design decision `rq1_series_level_test.py` made for RQ1.
"""
import numpy as np
import pandas as pd

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    fit_lgbm, fit_xgb, mase, N_LOCAL_SERIES, WEATHER_COLS, CALENDAR_COLS, PROMO_COLS,
)

RESULTS_DIR = "results"
GROUPS = {"weather": WEATHER_COLS, "calendar": CALENDAR_COLS, "promotion": PROMO_COLS}
SARIMAX_REGRESSORS = ["is_weekend", "is_public_holiday", "is_school_holiday",
                       "temp_anomaly", "is_promotion", "discount_pct"]


def per_series_mase(actual, pred, store_item, train_by_series):
    d = pd.DataFrame({"store_item": store_item, "actual": actual, "pred": pred})
    rows = []
    for sid, g in d.groupby("store_item"):
        rows.append({"store_item": sid,
                     "mase": mase(g["actual"].values, g["pred"].values,
                                  train_by_series.get_group(sid).values)})
    return pd.DataFrame(rows).set_index("store_item")["mase"]


def main():
    df = load_features()
    all_cols = feature_cols(df)
    train_mask, fit_mask, val_mask, cutoff = time_split(df)
    store_item_test = df.loc[~train_mask, "store_item"].values
    train_by_series = df.loc[train_mask].groupby("store_item")["sales"]

    variants = {"full": []}
    variants.update({f"no_{g}": cols for g, cols in GROUPS.items()})

    print("=== Global ablation: per-series MASE for LightGBM/XGBoost x 4 variants ===")
    wide = {}
    for variant, drop_cols in variants.items():
        cols = [c for c in all_cols if c not in drop_cols]
        X_fit, y_fit = df.loc[fit_mask, cols], df.loc[fit_mask, "sales"]
        X_val, y_val = df.loc[val_mask, cols], df.loc[val_mask, "sales"]
        X_test = df.loc[~train_mask, cols]
        cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]

        lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
        xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)
        wide[f"LightGBM_{variant}"] = per_series_mase(
            df.loc[~train_mask, "sales"].values, lgbm_model.predict(X_test), store_item_test, train_by_series)
        wide[f"XGBoost_{variant}"] = per_series_mase(
            df.loc[~train_mask, "sales"].values, xgb_model.predict(X_test), store_item_test, train_by_series)
        print(f"  {variant} done ({len(cols)} features)")

    global_per_series = pd.DataFrame(wide)
    global_per_series.to_csv(f"{RESULTS_DIR}/rq2_global_per_series_mase.csv")
    print(f"\nsaved {global_per_series.shape} to rq2_global_per_series_mase.csv")

    print("\n=== Local: per-series MASE for SARIMA (no regressors) vs SARIMAX (+regressors) ===")
    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    panel = df[["date", "store_item", "sales", "is_test"] + SARIMAX_REGRESSORS].copy()
    panel = panel[panel["store_item"].isin(local_series)].sort_values(["store_item", "date"])

    sarima_only = run_sarimax(panel, local_series, exog_cols=None)
    sarimax_full = run_sarimax(panel, local_series, exog_cols=SARIMAX_REGRESSORS)

    local_train_by_series = df.loc[train_mask].groupby("store_item")["sales"]
    local_rows = {}
    for label, preds in [("SARIMA", sarima_only), ("SARIMAX", sarimax_full)]:
        local_rows[label] = pd.Series({
            sid: mase(g["actual"].values, g["pred"].values, local_train_by_series.get_group(sid).values)
            for sid, g in preds.groupby("store_item")
        })
    local_per_series = pd.DataFrame(local_rows)
    local_per_series.index.name = "store_item"
    local_per_series.to_csv(f"{RESULTS_DIR}/rq2_local_per_series_mase.csv")
    print(f"saved {local_per_series.shape} to rq2_local_per_series_mase.csv")

    print("\nRQ2 series-level refit complete.")


if __name__ == "__main__":
    main()
