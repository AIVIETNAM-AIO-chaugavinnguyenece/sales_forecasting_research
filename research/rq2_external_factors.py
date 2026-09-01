"""RQ2 -- Do the external factors earn their place?

Research question: How much forecast accuracy do weather, calendar and
promotion regressors contribute, and does the contribution differ by model
family?
"""
import numpy as np
import pandas as pd

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    fit_lgbm, fit_xgb, wape, mase, N_LOCAL_SERIES, WEATHER_COLS, CALENDAR_COLS, PROMO_COLS,
)

RESULTS_DIR = "results"
GROUPS = {"weather": WEATHER_COLS, "calendar": CALENDAR_COLS, "promotion": PROMO_COLS}
# Same regressor set used for Prophet in 04_modelling_update.ipynb -- one
# representative numeric column per group, so SARIMAX gets a like-for-like
# regressor set to the global models' leave-one-group-out ablation.
SARIMAX_REGRESSORS = ["is_weekend", "is_public_holiday", "is_school_holiday",
                       "temp_anomaly", "is_promotion", "discount_pct"]


def fit_global(df, cols, fit_mask, val_mask, train_mask):
    X_fit, y_fit = df.loc[fit_mask, cols], df.loc[fit_mask, "sales"]
    X_val, y_val = df.loc[val_mask, cols], df.loc[val_mask, "sales"]
    X_test = df.loc[~train_mask, cols]
    cat_features = [c for c in cols if str(X_fit[c].dtype) == "category"]

    lgbm_model = fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features)
    xgb_model = fit_xgb(X_fit, y_fit, X_val, y_val)
    return {"LightGBM": lgbm_model.predict(X_test), "XGBoost": xgb_model.predict(X_test)}


def main():
    df = load_features()
    all_cols = feature_cols(df)
    train_mask, fit_mask, val_mask, cutoff = time_split(df)
    y_train_full = df.loc[train_mask, "sales"].values
    y_test = df.loc[~train_mask, "sales"].values

    print("=== Global models: leave-one-group-out ablation ===")
    variants = {"full": []}
    variants.update({f"no_{g}": cols for g, cols in GROUPS.items()})

    rows = []
    for variant, drop_cols in variants.items():
        cols = [c for c in all_cols if c not in drop_cols]
        preds = fit_global(df, cols, fit_mask, val_mask, train_mask)
        for m, p in preds.items():
            rows.append({"model": m, "variant": variant, "n_features": len(cols),
                         "MAE": np.mean(np.abs(y_test - p)),
                         "WAPE": wape(y_test, p),
                         "MASE": mase(y_test, p, y_train_full)})
        print(f"  {variant}: {len(cols)} features")

    global_results = pd.DataFrame(rows)
    full_wape = global_results[global_results["variant"] == "full"].set_index("model")["WAPE"]
    global_results["wape_cost_pct"] = global_results.apply(
        lambda r: 100 * (r["WAPE"] - full_wape[r["model"]]) / full_wape[r["model"]], axis=1)
    print("\n" + global_results.round(4).to_string(index=False))
    global_results.to_csv(f"{RESULTS_DIR}/rq2_global_ablation.csv", index=False)

    assert len(global_results) == 8, f"expected 8 rows (2 models x 4 variants), got {len(global_results)}"

    print("\n=== Local model: SARIMA (no regressors) vs SARIMAX (+ regressors) ===")
    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    panel = df[["date", "store_item", "sales", "is_test"] + SARIMAX_REGRESSORS].copy()
    panel = panel[panel["store_item"].isin(local_series)].sort_values(["store_item", "date"])

    print("SARIMA (no regressors):")
    sarima_only = run_sarimax(panel, local_series, exog_cols=None)
    print("SARIMAX (+regressors):")
    sarimax_full = run_sarimax(panel, local_series, exog_cols=SARIMAX_REGRESSORS)

    train_by_series = df.loc[train_mask].groupby("store_item")["sales"]
    local_rows = []
    for label, preds in [("SARIMA (no regressors)", sarima_only), ("SARIMAX (+regressors)", sarimax_full)]:
        per_series_mase = [
            mase(g["actual"].values, g["pred"].values, train_by_series.get_group(sid).values)
            for sid, g in preds.groupby("store_item")
        ]
        local_rows.append({
            "model": label,
            "MAE": np.mean(np.abs(preds["actual"] - preds["pred"])),
            "WAPE": wape(preds["actual"].values, preds["pred"].values),
            "MASE": np.nanmean(per_series_mase),
        })
    local_results = pd.DataFrame(local_rows)
    print("\n" + local_results.round(4).to_string(index=False))
    local_results.to_csv(f"{RESULTS_DIR}/rq2_local_regressor_value.csv", index=False)

    print("\nRQ2 evaluation complete.")


if __name__ == "__main__":
    main()
