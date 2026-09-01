"""RQ3 -- Attribution agreement and recovery of known drivers.

Research question: Do SHAP attributions from LightGBM and XGBoost agree with
each other, with permutation importance, and with SARIMAX coefficients -- and
do they recover the item-level drivers the generator actually used?
"""
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

from common import (
    load_features, feature_cols, time_split, select_local_series, run_sarimax,
    ITEM_TRUTH, SEED, N_LOCAL_SERIES,
)

RESULTS_DIR = "results"
N_EXPLAIN = 800
N_BACKGROUND = 60
TEMP_FEATURES = ["temp_anomaly", "temperature", "heat_excess", "cold_excess", "temp_norm"]
PRICE_FEATURES = ["log_price_ratio", "discount_pct", "price", "base_price", "is_deep_discount"]
SARIMAX_REGRESSORS = ["is_weekend", "is_public_holiday", "is_school_holiday",
                       "temp_anomaly", "is_promotion", "discount_pct"]

# SHAP's interventional TreeExplainer does not support XGBoost's native
# categorical splits (raises NotImplementedError). Following the precedent in
# notebooks/06_proof_EXECUTED_example.ipynb, both models here are fit on a
# numeric-coded feature matrix with a lighter tree count, purpose-built for
# the attribution study -- not the same model objects RQ1/RQ2 forecast with.
N_ESTIMATORS_SHAP = 300


def numeric_matrix(df, cols):
    X = df[cols].copy()
    for c in cols:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.astype("int32")
    return X.astype("float32")


def fit_global(X_train, y_train):
    import lightgbm as lgbm
    import xgboost as xgb
    lgbm_model = lgbm.LGBMRegressor(
        n_estimators=N_ESTIMATORS_SHAP, num_leaves=63, learning_rate=0.05,
        min_child_samples=40, random_state=SEED, n_jobs=-1, verbose=-1)
    lgbm_model.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS_SHAP, learning_rate=0.05, max_depth=8,
        min_child_weight=10, tree_method="hist", enable_categorical=False,
        random_state=SEED, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train)
    return lgbm_model, xgb_model


def shap_importance(model, X_train, X_test, cols, seed):
    rng = np.random.default_rng(seed)
    n_test = len(X_test)
    explain_idx = np.sort(rng.choice(n_test, min(N_EXPLAIN, n_test), replace=False))
    bg_idx = rng.choice(len(X_train), N_BACKGROUND, replace=False)
    X_ex = X_test.iloc[explain_idx]
    background = X_train.iloc[bg_idx]
    explainer = shap.TreeExplainer(
        model, data=shap.maskers.Independent(background, max_samples=N_BACKGROUND),
        feature_perturbation="interventional",
    )
    sv = np.asarray(explainer.shap_values(X_ex, check_additivity=False))
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols)
    return mean_abs, sv, explain_idx


def fit_sarimax_coefficients(panel, series_ids):
    rows = []
    for sid in series_ids:
        g = panel[panel["store_item"] == sid]
        tr = g[~g["is_test"]].reset_index(drop=True)
        if len(tr) < 60:
            continue
        item_id = int(tr["item_id"].iloc[0])
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            m = SARIMAX(tr["sales"].values, exog=tr[SARIMAX_REGRESSORS], order=(1, 1, 1),
                        seasonal_order=(1, 0, 1, 7), enforce_stationarity=False,
                        enforce_invertibility=False)
            fit = m.fit(disp=False, maxiter=100)
            # Named exog -> fit.param_names carries the real regressor names.
            # Extracting by name (not position) matters: parameter order shifts
            # with the trend/seasonal spec, so a positional slice would silently
            # pull the wrong coefficients.
            params = dict(zip(fit.param_names, fit.params))
            coef = {c: params.get(c, np.nan) for c in SARIMAX_REGRESSORS}
        except Exception as e:
            print(f"    [skip] {sid}: {type(e).__name__}: {e}")
            coef = {c: np.nan for c in SARIMAX_REGRESSORS}
        rows.append({"store_item": sid, "item_id": item_id, **coef})
    return pd.DataFrame(rows)


def main():
    df = load_features()
    cols = feature_cols(df)
    train_mask, fit_mask, val_mask, cutoff = time_split(df)

    X_all = numeric_matrix(df, cols)
    X_train = X_all.loc[train_mask]
    X_test = X_all.loc[~train_mask]
    y_train = df.loc[train_mask, "sales"]
    y_test = df.loc[~train_mask, "sales"]
    item_ids_test = df.loc[~train_mask, "item_id"].values

    print("=== Fitting global models (numeric-coded features, for SHAP compatibility) ===")
    lgbm_model, xgb_model = fit_global(X_train, y_train)

    print("\n=== SHAP importance (mean |SHAP|, same explain/background sample for both models) ===")
    shap_lgbm, sv_lgbm, idx = shap_importance(lgbm_model, X_train, X_test, cols, SEED)
    shap_xgb, sv_xgb, idx_check = shap_importance(xgb_model, X_train, X_test, cols, SEED)
    assert np.array_equal(idx, idx_check), "explain sample must be identical across models"

    rho_shap, p_shap = stats.spearmanr(shap_lgbm, shap_xgb)
    print(f"  LightGBM vs XGBoost feature-ranking agreement: rho={rho_shap:.3f} (p={p_shap:.4g})")

    print("\n=== Permutation importance (test-set MAE increase, 3,000-row sample) ===")
    sample_idx = np.random.default_rng(SEED).choice(len(X_test), min(3000, len(X_test)), replace=False)
    X_perm, y_perm = X_test.iloc[sample_idx], y_test.iloc[sample_idx]
    perm_lgbm = permutation_importance(lgbm_model, X_perm, y_perm, n_repeats=5,
                                        random_state=SEED, scoring="neg_mean_absolute_error")
    perm_xgb = permutation_importance(xgb_model, X_perm, y_perm, n_repeats=5,
                                       random_state=SEED, scoring="neg_mean_absolute_error")
    perm_lgbm_s = pd.Series(perm_lgbm.importances_mean, index=cols)
    perm_xgb_s = pd.Series(perm_xgb.importances_mean, index=cols)

    rho_perm_lgbm, _ = stats.spearmanr(shap_lgbm, perm_lgbm_s)
    rho_perm_xgb, _ = stats.spearmanr(shap_xgb, perm_xgb_s)
    print(f"  LightGBM: SHAP vs permutation importance rho={rho_perm_lgbm:.3f}")
    print(f"  XGBoost : SHAP vs permutation importance rho={rho_perm_xgb:.3f}")

    agreement = pd.DataFrame([
        {"comparison": "SHAP LightGBM vs SHAP XGBoost", "rho": rho_shap, "p_value": p_shap},
        {"comparison": "LightGBM: SHAP vs permutation", "rho": rho_perm_lgbm, "p_value": np.nan},
        {"comparison": "XGBoost: SHAP vs permutation", "rho": rho_perm_xgb, "p_value": np.nan},
    ])
    print("\n" + agreement.round(4).to_string(index=False))
    agreement.to_csv(f"{RESULTS_DIR}/rq3_attribution_agreement.csv", index=False)

    print("\n=== SARIMAX coefficients (price & temperature regressors, 24-series sample) ===")
    local_series = select_local_series(df, train_mask, n=N_LOCAL_SERIES)
    panel = df[["date", "store_item", "item_id", "sales", "is_test"] + SARIMAX_REGRESSORS].copy()
    panel = panel[panel["store_item"].isin(local_series)].sort_values(["store_item", "date"])
    coefs = fit_sarimax_coefficients(panel, local_series)
    coefs.to_csv(f"{RESULTS_DIR}/rq3_sarimax_coefficients.csv", index=False)
    print(coefs.round(4).to_string(index=False))

    print("\n=== Recovery of known item-level drivers (Spearman rho vs generator ground truth) ===")
    recovery_rows = []
    for driver, features, truth_col in [
        ("temperature", TEMP_FEATURES, "abs_temp_effect"),
        ("price", PRICE_FEATURES, "abs_elasticity"),
    ]:
        idx_f = [cols.index(c) for c in features if c in cols]
        for model_name, sv in [("LightGBM", sv_lgbm), ("XGBoost", sv_xgb)]:
            mass = pd.Series(np.abs(sv[:, idx_f]).sum(axis=1)).groupby(item_ids_test[idx]).mean()
            merged = mass.reset_index()
            merged.columns = ["item_id", "shap_mass"]
            merged = merged.merge(ITEM_TRUTH, on="item_id")
            rho, p = stats.spearmanr(merged["shap_mass"], merged[truth_col])
            recovery_rows.append({"driver": driver, "method": f"SHAP {model_name}", "rho": rho, "p_value": p})

        coef_col = "temp_anomaly" if driver == "temperature" else "discount_pct"
        sar = coefs.groupby("item_id")[coef_col].apply(lambda s: s.abs().mean()).reset_index()
        sar.columns = ["item_id", "coef_mass"]
        sar = sar.merge(ITEM_TRUTH, on="item_id")
        rho, p = stats.spearmanr(sar["coef_mass"], sar[truth_col])
        recovery_rows.append({"driver": driver, "method": "SARIMAX coefficient", "rho": rho, "p_value": p})

    recovery = pd.DataFrame(recovery_rows)
    print(recovery.round(4).to_string(index=False))
    recovery.to_csv(f"{RESULTS_DIR}/rq3_recovery.csv", index=False)

    assert len(recovery) == 6, f"expected 6 rows (2 drivers x 3 methods), got {len(recovery)}"
    assert recovery.loc[recovery["method"].str.startswith("SHAP"), "rho"].notna().all(), \
        "SHAP-based recovery rho must not be NaN"

    print("\nRQ3 evaluation complete.")


if __name__ == "__main__":
    main()
