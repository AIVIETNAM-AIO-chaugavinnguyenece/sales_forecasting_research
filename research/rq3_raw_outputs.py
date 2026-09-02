"""RQ3 addendum -- persist raw per-feature/per-item outputs for a real audit.

`rq3_attribution_agreement.py` only saves the summary rho/p (`rq3_attribution_
agreement.csv`, `rq3_recovery.csv`) -- not the underlying 62-feature SHAP /
permutation importance vectors or the per-item attribution mass. Without those,
a principal-level audit can't (a) check whether Pearson vs Spearman actually
matters here, (b) check whether SHAP/permutation and SARIMAX outputs are even
on comparable footing, or (c) compute recall/precision@k against the generator
ground truth. This script re-fits the same models with the same seed as
`rq3_attribution_agreement.py` and persists those raw vectors.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from common import load_features, feature_cols, time_split, ITEM_TRUTH, SEED

RESULTS_DIR = "results"
N_EXPLAIN = 800
N_BACKGROUND = 60
TEMP_FEATURES = ["temp_anomaly", "temperature", "heat_excess", "cold_excess", "temp_norm"]
PRICE_FEATURES = ["log_price_ratio", "discount_pct", "price", "base_price", "is_deep_discount"]
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

    print("=== Fitting global models ===")
    lgbm_model, xgb_model = fit_global(X_train, y_train)

    print("=== SHAP importance ===")
    shap_lgbm, sv_lgbm, idx = shap_importance(lgbm_model, X_train, X_test, cols, SEED)
    shap_xgb, sv_xgb, idx_check = shap_importance(xgb_model, X_train, X_test, cols, SEED)
    assert np.array_equal(idx, idx_check)

    print("=== Permutation importance ===")
    sample_idx = np.random.default_rng(SEED).choice(len(X_test), min(3000, len(X_test)), replace=False)
    X_perm, y_perm = X_test.iloc[sample_idx], y_test.iloc[sample_idx]
    perm_lgbm = permutation_importance(lgbm_model, X_perm, y_perm, n_repeats=5,
                                        random_state=SEED, scoring="neg_mean_absolute_error")
    perm_xgb = permutation_importance(xgb_model, X_perm, y_perm, n_repeats=5,
                                       random_state=SEED, scoring="neg_mean_absolute_error")

    raw = pd.DataFrame({
        "feature": cols,
        "shap_LightGBM": shap_lgbm.values,
        "shap_XGBoost": shap_xgb.values,
        "perm_LightGBM": perm_lgbm.importances_mean,
        "perm_XGBoost": perm_xgb.importances_mean,
    }).set_index("feature")
    raw.to_csv(f"{RESULTS_DIR}/rq3_raw_importances.csv")
    print(f"saved {raw.shape} to rq3_raw_importances.csv")

    print("=== Per-item SHAP attribution mass (temperature, price) ===")
    item_rows = []
    for driver, features in [("temperature", TEMP_FEATURES), ("price", PRICE_FEATURES)]:
        idx_f = [cols.index(c) for c in features if c in cols]
        for model_name, sv in [("LightGBM", sv_lgbm), ("XGBoost", sv_xgb)]:
            mass = pd.Series(np.abs(sv[:, idx_f]).sum(axis=1)).groupby(item_ids_test[idx]).mean()
            for item_id, val in mass.items():
                item_rows.append({"item_id": item_id, "driver": driver, "method": f"SHAP_{model_name}",
                                   "mass": val})
    item_mass = pd.DataFrame(item_rows)
    item_mass = item_mass.merge(ITEM_TRUTH, on="item_id")
    item_mass.to_csv(f"{RESULTS_DIR}/rq3_item_mass_raw.csv", index=False)
    print(f"saved {item_mass.shape} to rq3_item_mass_raw.csv")

    print("\nRQ3 raw-outputs refit complete.")


if __name__ == "__main__":
    main()
