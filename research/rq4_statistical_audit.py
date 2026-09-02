"""RQ4 principal-level audit -- selecting-statistical-test, checking-test-
assumptions, reporting-effect-sizes, applied to the real per-series MASE
deltas from `rq4_series_level_test.py`.

Two distinct questions, two distinct designs:
  - "Does transform T help model M?" -- paired (same series, T vs raw).
  - "Does T's effect differ between global (LightGBM/XGBoost) and local
    (SARIMA) families?" -- an interaction, tested as a difference-of-
    differences restricted to the 24 series both families were scored on
    (the same restriction RQ1's cross-family comparison used), not a
    side-by-side comparison of two aggregate WAPE-delta percentages.
Multiple-comparison families are locked before testing: the 6 main-effect
tests (2 transforms x 3 models) are one family; the 4 interaction tests
(2 transforms x 2 family-pairs) are a second family.
"""
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
GLOBAL_MODELS = ["LightGBM", "XGBoost"]
VARIANTS = ["log1p", "zscore"]


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


def paired_test(a, b, label):
    """Gate on Shapiro-Wilk of the paired difference, then run the test it selects."""
    valid = a.notna() & b.notna()
    if not valid.all():
        print(f"  [{label}] dropping {(~valid).sum()} series with NaN MASE")
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


def main():
    global_ps = pd.read_csv(f"{RESULTS_DIR}/rq4_global_per_series_mase.csv").set_index("store_item")
    local_ps = pd.read_csv(f"{RESULTS_DIR}/rq4_local_per_series_mase.csv").set_index("store_item")

    print("=== Main effect: transform vs raw, within each model family (paired over series) ===")
    main_rows = []
    for model in GLOBAL_MODELS:
        for variant in VARIANTS:
            res = paired_test(global_ps[f"{model}_{variant}"], global_ps[f"{model}_raw"],
                               f"{model}: {variant} vs raw")
            res.update({"model": model, "variant": variant, "n_series_pool": "global (360)"})
            main_rows.append(res)
    for variant in VARIANTS:
        res = paired_test(local_ps[variant], local_ps["raw"], f"SARIMA: {variant} vs raw")
        res.update({"model": "SARIMA", "variant": variant, "n_series_pool": "local (24)"})
        main_rows.append(res)

    main_df = pd.DataFrame(main_rows)
    reject, adj_alpha = holm_bonferroni(main_df["p_raw"].tolist())
    main_df["holm_adjusted_alpha"] = adj_alpha
    main_df["significant_after_holm"] = reject
    print(main_df[["model", "variant", "n", "normal", "test", "p_raw", "holm_adjusted_alpha",
                    "significant_after_holm", "median_delta", "effect_name", "effect",
                    "effect_ci_lo", "effect_ci_hi"]].round(4).to_string(index=False))
    main_df.to_csv(f"{RESULTS_DIR}/rq4_main_effect_tests.csv", index=False)

    print("\n=== Interaction: does the transform effect differ between global and local families? ===")
    print("(restricted to the 24 series both families were scored on; difference-of-differences)")
    local_series = local_ps.index
    inter_rows = []
    for variant in VARIANTS:
        delta_sarima = local_ps[variant] - local_ps["raw"]
        for model in GLOBAL_MODELS:
            delta_model = (global_ps.loc[local_series, f"{model}_{variant}"]
                           - global_ps.loc[local_series, f"{model}_raw"])
            res = paired_test(delta_model, delta_sarima, f"interaction: {variant}, {model} vs SARIMA")
            res.update({"variant": variant, "family_pair": f"{model} vs SARIMA"})
            inter_rows.append(res)

    inter_df = pd.DataFrame(inter_rows)
    reject_i, adj_alpha_i = holm_bonferroni(inter_df["p_raw"].tolist())
    inter_df["holm_adjusted_alpha"] = adj_alpha_i
    inter_df["significant_after_holm"] = reject_i
    print(inter_df[["variant", "family_pair", "n", "normal", "test", "p_raw", "holm_adjusted_alpha",
                     "significant_after_holm", "median_delta", "effect_name", "effect",
                     "effect_ci_lo", "effect_ci_hi"]].round(4).to_string(index=False))
    inter_df.to_csv(f"{RESULTS_DIR}/rq4_interaction_tests.csv", index=False)

    print("\n=== Empirical check: is per-series z-score really a no-op for SARIMA? ===")
    zscore_check = paired_test(local_ps["zscore"], local_ps["raw"], "SARIMA: zscore vs raw (no-op check)")
    print({k: v for k, v in zscore_check.items() if k != "comparison"})
    pd.DataFrame([zscore_check]).to_csv(f"{RESULTS_DIR}/rq4_sarima_zscore_noop_check.csv", index=False)

    print("\nRQ4 statistical audit complete.")


if __name__ == "__main__":
    main()
