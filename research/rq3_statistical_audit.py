"""RQ3 principal-level audit -- selecting-statistical-test, checking-test-
assumptions, reporting-effect-sizes, applied to the raw importance vectors
from `rq3_raw_outputs.py`.

Three distinct comparisons are in play, and they are NOT interchangeable:
  1. SHAP-LightGBM vs SHAP-XGBoost, and SHAP vs permutation (per model) --
     same 62-feature universe, same underlying test rows. A same-footing
     rank-agreement question.
  2. SHAP / permutation vs SARIMAX coefficients -- SARIMAX only has 6 named
     exogenous regressors (it structurally excludes the ~20 lag/rolling/EWMA
     columns SHAP/permutation see, by RQ3's own lag-absorption design). A
     "full feature set" agreement test is undefined here; the only honest
     comparison restricts to the 6 shared regressor names, which leaves n=6
     -- too small for a trustworthy correlation, and the audit says so
     instead of reporting a misleadingly precise number.
  3. Item-level driver recovery (SHAP mass / SARIMAX coefficient vs the
     generator's held-out per-item ground truth) -- n=30 items, already
     flagged elsewhere in this codebase as fragile at that n.
"""
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
SARIMAX_REGRESSORS = ["is_weekend", "is_public_holiday", "is_school_holiday",
                       "temp_anomaly", "is_promotion", "discount_pct"]


def spearman_ci(rho, n, alpha=0.05):
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(rho)
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def report_pair(name, x, y, n_min_reliable=10):
    n = len(x)
    r_pearson, p_pearson = stats.pearsonr(x, y)
    rho, p_spear = stats.spearmanr(x, y)
    row = {"pair": name, "n": n, "pearson_r": r_pearson, "pearson_p": p_pearson,
           "spearman_rho": rho, "spearman_p": p_spear}
    if n >= n_min_reliable + 3:
        lo, hi = spearman_ci(rho, n)
        row["spearman_ci_lo"], row["spearman_ci_hi"] = lo, hi
        row["reliable_n"] = True
    else:
        tau, p_tau = stats.kendalltau(x, y)
        row["kendall_tau"], row["kendall_p"] = tau, p_tau
        row["spearman_ci_lo"], row["spearman_ci_hi"] = np.nan, np.nan
        row["reliable_n"] = False
    return row


def main():
    raw = pd.read_csv(f"{RESULTS_DIR}/rq3_raw_importances.csv").set_index("feature")

    print("=== Step 1/2: distribution shape of each importance output (n=62 features) ===")
    shape_rows = []
    for col in raw.columns:
        v = raw[col].values
        w, p = stats.shapiro(v)
        shape_rows.append({"method": col, "shapiro_W": w, "shapiro_p": p,
                            "skew": stats.skew(v), "normal_at_05": p > 0.05})
    shape_df = pd.DataFrame(shape_rows)
    print(shape_df.round(4).to_string(index=False))
    shape_df.to_csv(f"{RESULTS_DIR}/rq3_importance_shape.csv", index=False)

    print("\n=== Step 1/3: same-footing agreement (n=62 features, Pearson vs Spearman) ===")
    same_footing_rows = [
        report_pair("SHAP LightGBM vs SHAP XGBoost", raw["shap_LightGBM"], raw["shap_XGBoost"]),
        report_pair("LightGBM: SHAP vs permutation", raw["shap_LightGBM"], raw["perm_LightGBM"]),
        report_pair("XGBoost: SHAP vs permutation", raw["shap_XGBoost"], raw["perm_XGBoost"]),
    ]
    same_footing = pd.DataFrame(same_footing_rows)
    print(same_footing.round(4).to_string(index=False))
    same_footing.to_csv(f"{RESULTS_DIR}/rq3_same_footing_agreement.csv", index=False)

    print("\n=== Step 2: comparable-footing check -- SHAP/permutation (62 features) vs SARIMAX (6 regressors) ===")
    missing = [c for c in SARIMAX_REGRESSORS if c not in raw.index]
    print(f"SARIMAX regressor set: {SARIMAX_REGRESSORS}")
    print(f"Present in the 62-feature SHAP/permutation universe: {[c for c in SARIMAX_REGRESSORS if c in raw.index]}")
    if missing:
        print(f"NOT present (SARIMAX-only or renamed): {missing}")
    print("SARIMAX structurally excludes the ~20 lag/rolling/EWMA columns (by RQ3's own lag-absorption")
    print("design) plus store/item identifiers and most calendar dummies -- a 62-feature-universe")
    print("agreement test against SARIMAX is undefined. Restricting to the 6 shared regressors:")

    sarimax_coefs = pd.read_csv(f"{RESULTS_DIR}/rq3_sarimax_coefficients.csv")
    shared = [c for c in SARIMAX_REGRESSORS if c in raw.index]
    sarimax_mean_abs = sarimax_coefs[shared].abs().mean()

    restricted_rows = []
    for imp_col in ["shap_LightGBM", "shap_XGBoost", "perm_LightGBM", "perm_XGBoost"]:
        x = raw.loc[shared, imp_col].values
        y = sarimax_mean_abs.loc[shared].values
        row = report_pair(f"{imp_col} vs SARIMAX |coef| (n=6 shared regressors)", x, y)
        restricted_rows.append(row)
    restricted = pd.DataFrame(restricted_rows)
    print(restricted.round(4).to_string(index=False))
    print(f"\nn=6 is well below the ~10-observation floor for a trustworthy correlation -- these numbers")
    print("are reported for transparency only and should NOT be read as a confirmed agreement/disagreement.")
    restricted.to_csv(f"{RESULTS_DIR}/rq3_restricted_footing_agreement.csv", index=False)

    print("\n=== Step 3: item-level driver recovery, with CI this time (n=30 items) ===")
    item_mass = pd.read_csv(f"{RESULTS_DIR}/rq3_item_mass_raw.csv")
    recovery_rows = []
    for driver, truth_col in [("temperature", "abs_temp_effect"), ("price", "abs_elasticity")]:
        for method in ["SHAP_LightGBM", "SHAP_XGBoost"]:
            sub = item_mass[(item_mass["driver"] == driver) & (item_mass["method"] == method)]
            rho, p = stats.spearmanr(sub["mass"], sub[truth_col])
            lo, hi = spearman_ci(rho, len(sub))
            recovery_rows.append({"driver": driver, "method": method, "n": len(sub),
                                   "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi})
        coef_col = "temp_anomaly" if driver == "temperature" else "discount_pct"
        sar = sarimax_coefs.groupby("item_id")[coef_col].apply(lambda s: s.abs().mean())
        # ITEM_TRUTH values reused from the item_mass table (already merged there per driver/method)
        truth = item_mass[(item_mass["driver"] == driver) & (item_mass["method"] == "SHAP_LightGBM")][
            ["item_id", truth_col]].set_index("item_id")[truth_col]
        merged = pd.concat([sar.rename("coef_mass"), truth], axis=1).dropna()
        rho, p = stats.spearmanr(merged["coef_mass"], merged[truth_col])
        lo, hi = spearman_ci(rho, len(merged))
        recovery_rows.append({"driver": driver, "method": "SARIMAX_coefficient", "n": len(merged),
                               "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi})
    recovery = pd.DataFrame(recovery_rows)
    print(recovery.round(4).to_string(index=False))
    recovery.to_csv(f"{RESULTS_DIR}/rq3_recovery_with_ci.csv", index=False)

    print("\n=== Step 4: recall/precision@k against generator ground truth ===")
    topk_rows = []
    for driver, truth_col in [("temperature", "abs_temp_effect"), ("price", "abs_elasticity")]:
        truth_sub = item_mass[(item_mass["driver"] == driver) & (item_mass["method"] == "SHAP_LightGBM")][
            ["item_id", truth_col]].set_index("item_id")[truth_col]
        for k in [5, 10]:
            true_top = set(truth_sub.nlargest(k).index)
            for method in ["SHAP_LightGBM", "SHAP_XGBoost"]:
                sub = item_mass[(item_mass["driver"] == driver) & (item_mass["method"] == method)].set_index("item_id")["mass"]
                pred_top = set(sub.nlargest(k).index)
                overlap = len(true_top & pred_top)
                topk_rows.append({"driver": driver, "method": method, "k": k,
                                   "recall_at_k": overlap / k, "precision_at_k": overlap / k,
                                   "n_overlap": overlap})
            coef_col = "temp_anomaly" if driver == "temperature" else "discount_pct"
            sar = sarimax_coefs.groupby("item_id")[coef_col].apply(lambda s: s.abs().mean())
            pred_top = set(sar.nlargest(k).index)
            overlap = len(true_top & pred_top)
            topk_rows.append({"driver": driver, "method": "SARIMAX_coefficient", "k": k,
                               "recall_at_k": overlap / k, "precision_at_k": overlap / k,
                               "n_overlap": overlap})
    topk = pd.DataFrame(topk_rows)
    print(topk.round(3).to_string(index=False))
    topk.to_csv(f"{RESULTS_DIR}/rq3_topk_recovery.csv", index=False)

    print("\nRQ3 statistical audit complete.")


if __name__ == "__main__":
    main()
