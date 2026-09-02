"""RQ2 principal-level audit -- selecting-statistical-test, checking-test-
assumptions, reporting-effect-sizes, applied to the real per-series deltas
from `rq2_series_level_test.py`.

Design, stated explicitly instead of assumed:
  - "Does regressor group G help model M?" is a PAIRED design (same series,
    with vs without G) -- not two independent samples. Gate: Shapiro-Wilk on
    the per-series delta; Normal -> paired t-test, else Wilcoxon signed-rank.
  - "Does G's contribution differ between LightGBM and XGBoost?" is an
    INTERACTION question. Collapsing it into "compare LightGBM's WAPE-cost to
    XGBoost's WAPE-cost" throws away the pairing (both deltas share the same
    360 series) and can't distinguish a real interaction from folder-level
    noise. The correct move is a difference-of-differences per series --
    diff_i = delta_LightGBM_i - delta_XGBoost_i -- tested against zero with
    the same paired-test gate. This is the repeated-measures analogue of a
    2x2 (model family x with/without) interaction contrast.
Multiple-comparison families are locked before testing (comparing-models-
fairly discipline): the 6 main-effect tests (3 groups x 2 models) are one
family: the 3 interaction tests are a second family. Each gets its own
Holm-Bonferroni correction; they are not one pool of 9.
"""
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
GROUPS = ["weather", "calendar", "promotion"]
MODELS = ["LightGBM", "XGBoost"]


def holm_bonferroni(pvalues, alpha=0.05):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    reject = [False] * m
    adj_alpha = [None] * m
    for rank, idx in enumerate(order):
        a = alpha / (m - rank)
        adj_alpha[idx] = a
        if pvalues[idx] < a:
            reject[idx] = True
        else:
            break
    return reject, adj_alpha


def paired_test(a, b, label):
    """Gate on Shapiro-Wilk of the paired difference, then run the test it selects."""
    valid = a.notna() & b.notna()
    if not valid.all():
        print(f"  [{label}] dropping {(~valid).sum()} series with NaN MASE (degenerate naive denominator)")
    a, b = a[valid], b[valid]
    d = a - b
    w, p_norm = stats.shapiro(d)
    normal = p_norm > 0.05
    if normal:
        stat, p = stats.ttest_rel(a, b)
        test_name = "paired t-test"
        n = len(d)
        dz = d.mean() / d.std(ddof=1)
        se_dz = np.sqrt(1 / n + dz**2 / (2 * n))
        eff, eff_lo, eff_hi = dz, dz - 1.96 * se_dz, dz + 1.96 * se_dz
        eff_name = "Cohen's dz"
    else:
        stat, p = stats.wilcoxon(a, b)
        test_name = "Wilcoxon signed-rank"
        eff, eff_lo, eff_hi = rank_biserial_ci(a.values, b.values)
        eff_name = "matched-pairs rank-biserial r"
    return {
        "comparison": label, "n": len(d), "shapiro_p": p_norm, "normal": normal,
        "test": test_name, "stat": stat, "p_raw": p,
        "median_delta": np.median(d), "mean_delta": d.mean(),
        "effect_name": eff_name, "effect": eff, "effect_ci_lo": eff_lo, "effect_ci_hi": eff_hi,
    }


def rank_biserial(a, b):
    d = a - b
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    w_pos, w_neg = r[d > 0].sum(), r[d < 0].sum()
    return 1 - 4 * min(w_pos, w_neg) / (n * (n + 1))


def rank_biserial_ci(a, b, n_boot=3000, seed=2025):
    rng = np.random.default_rng(seed)
    n = len(a)
    r = rank_biserial(a, b)
    boot = [rank_biserial(a[idx], b[idx]) for idx in
            (rng.choice(n, n, replace=True) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return r, lo, hi


def main():
    per_series = pd.read_csv(f"{RESULTS_DIR}/rq2_global_per_series_mase.csv").set_index("store_item")

    print("=== Main effect: with-vs-without, per model family (paired over series) ===")
    main_rows = []
    for model in MODELS:
        for group in GROUPS:
            full = per_series[f"{model}_full"]
            no_group = per_series[f"{model}_no_{group}"]
            res = paired_test(no_group, full, f"{model}: no_{group} vs full")
            res.update({"model": model, "group": group})
            main_rows.append(res)
    main_df = pd.DataFrame(main_rows)

    reject, adj_alpha = holm_bonferroni(main_df["p_raw"].tolist())
    main_df["holm_adjusted_alpha"] = adj_alpha
    main_df["significant_after_holm"] = reject
    print(main_df[["model", "group", "n", "normal", "test", "p_raw", "holm_adjusted_alpha",
                    "significant_after_holm", "median_delta", "effect_name", "effect",
                    "effect_ci_lo", "effect_ci_hi"]].round(4).to_string(index=False))
    main_df.to_csv(f"{RESULTS_DIR}/rq2_main_effect_tests.csv", index=False)

    print("\n=== Interaction: does the group's contribution differ by model family? ===")
    print("(difference-of-differences per series: delta_LightGBM_i - delta_XGBoost_i, tested against 0)")
    inter_rows = []
    for group in GROUPS:
        delta_lgbm = per_series[f"LightGBM_no_{group}"] - per_series[f"LightGBM_full"]
        delta_xgb = per_series[f"XGBoost_no_{group}"] - per_series[f"XGBoost_full"]
        res = paired_test(delta_lgbm, delta_xgb, f"interaction: {group} (LightGBM delta vs XGBoost delta)")
        res["group"] = group
        inter_rows.append(res)
    inter_df = pd.DataFrame(inter_rows)

    reject_i, adj_alpha_i = holm_bonferroni(inter_df["p_raw"].tolist())
    inter_df["holm_adjusted_alpha"] = adj_alpha_i
    inter_df["significant_after_holm"] = reject_i
    print(inter_df[["group", "n", "normal", "test", "p_raw", "holm_adjusted_alpha",
                     "significant_after_holm", "median_delta", "effect_name", "effect",
                     "effect_ci_lo", "effect_ci_hi"]].round(4).to_string(index=False))
    inter_df.to_csv(f"{RESULTS_DIR}/rq2_interaction_tests.csv", index=False)

    print("\n=== Local model: SARIMA (no regressors) vs SARIMAX (+regressors), paired over 24 series ===")
    local = pd.read_csv(f"{RESULTS_DIR}/rq2_local_per_series_mase.csv", index_col=0)
    local_res = paired_test(local["SARIMA"], local["SARIMAX"], "SARIMA vs SARIMAX")
    local_df = pd.DataFrame([local_res])
    print(local_df[["n", "normal", "test", "p_raw", "median_delta", "effect_name", "effect",
                     "effect_ci_lo", "effect_ci_hi"]].round(4).to_string(index=False))
    local_df.to_csv(f"{RESULTS_DIR}/rq2_local_test.csv", index=False)

    print("\nRQ2 statistical audit complete.")


if __name__ == "__main__":
    main()
