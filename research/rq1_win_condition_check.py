"""RQ1 addendum -- does the original 'which series conditions does each model
win under' claim survive a formal test, or was it read off descriptive means?

`RQ1_report.md`'s original segmented-MASE table only reports band-level means
(volume_band, cv_band) with no test or CI -- exactly the "X wins" / eyeballed-
band framing `ml-datasci/reporting-effect-sizes` refuses to accept. This script
tests, on the real per-series MASE table from `rq1_series_level_test.py`,
whether the model gap actually correlates with series conditions (Spearman,
matching the rank-based-vs-Pearson lesson already applied elsewhere in this
project -- series volume/zero-rate are skewed, not Normal), whether the gap
differs across the pre-existing volume/cv bands (Kruskal-Wallis, since the
Shapiro check in `rq1_series_level_test.py` already ruled out Normality for
this kind of per-series metric), and reports the win-margin effect size
(matched-pairs rank-biserial, bootstrap 95% CI) instead of a bare "N/24 wins"
count.
"""
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
SEED = 2025


def spearman_ci(rho, n, alpha=0.05):
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(rho)
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def rank_biserial(a, b):
    d = a - b
    d = d[d != 0]
    n = len(d)
    r = stats.rankdata(np.abs(d))
    w_pos, w_neg = r[d > 0].sum(), r[d < 0].sum()
    return 1 - 4 * min(w_pos, w_neg) / (n * (n + 1))


def bootstrap_rank_biserial_ci(a, b, n_boot=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a)
    boot = [rank_biserial(a[idx], b[idx]) for idx in
            (rng.choice(n, n, replace=True) for _ in range(n_boot))]
    return np.nanpercentile(boot, [2.5, 97.5])


def main():
    per_series = pd.read_csv(f"{RESULTS_DIR}/rq1_per_series_mase.csv").set_index("store_item")
    cond = pd.read_csv(f"{RESULTS_DIR}/rq1_series_conditions.csv").set_index("store_item")
    m = per_series.join(cond[["mean_volume", "cv", "zero_rate", "cv_band", "volume_band"]])
    m["gap_sarima_minus_lgbm"] = m["SARIMA"] - m["LightGBM"]
    m["gap_xgb_minus_lgbm"] = m["XGBoost"] - m["LightGBM"]

    print("=== Correlation between the LightGBM-vs-SARIMA gap and series conditions (Spearman, n=24) ===")
    corr_rows = []
    for var in ["mean_volume", "cv", "zero_rate"]:
        rho, p = stats.spearmanr(m[var], m["gap_sarima_minus_lgbm"])
        lo, hi = spearman_ci(rho, len(m))
        corr_rows.append({"gap": "SARIMA-LightGBM", "condition": var, "rho": rho,
                           "ci_lo": lo, "ci_hi": hi, "p": p})
        print(f"  gap vs {var}: rho={rho:+.3f} 95% CI=[{lo:+.3f},{hi:+.3f}] p={p:.4f}")

    print("\n=== Kruskal-Wallis: does the SARIMA-LightGBM gap differ across the report's own bands? ===")
    kw_rows = []
    for band_col in ["volume_band", "cv_band"]:
        groups = [g["gap_sarima_minus_lgbm"].values for _, g in m.groupby(band_col, observed=True)]
        h, p = stats.kruskal(*groups)
        kw_rows.append({"band": band_col, "H": h, "p": p, "n_per_group": [len(g) for g in groups]})
        print(f"  {band_col}: H={h:.3f} p={p:.4f} n_per_group={[len(g) for g in groups]}")

    print("\n=== Win-margin effect size: matched-pairs rank-biserial (bootstrap 95% CI over series) ===")
    rb_rows = []
    for a, b in [("LightGBM", "XGBoost"), ("LightGBM", "SARIMA"), ("XGBoost", "SARIMA")]:
        r = rank_biserial(m[a].values, m[b].values)
        lo, hi = bootstrap_rank_biserial_ci(m[a].values, m[b].values)
        n_wins = int((m[a] < m[b]).sum())
        rb_rows.append({"pair": f"{a} vs {b}", "rank_biserial_r": r, "ci_lo": lo, "ci_hi": hi,
                         "n_wins": n_wins, "n_total": len(m)})
        print(f"  {a} vs {b}: r_rb={r:.3f} 95% CI=[{lo:.3f},{hi:.3f}]  "
              f"({a} has lower MASE in {n_wins}/{len(m)} series)")

    pd.DataFrame(corr_rows).to_csv(f"{RESULTS_DIR}/rq1_gap_condition_correlation.csv", index=False)
    pd.DataFrame(kw_rows).to_csv(f"{RESULTS_DIR}/rq1_gap_kruskal_wallis.csv", index=False)
    pd.DataFrame(rb_rows).to_csv(f"{RESULTS_DIR}/rq1_win_margin_effect_sizes.csv", index=False)
    m.to_csv(f"{RESULTS_DIR}/rq1_series_level_with_conditions.csv")

    print("\nRQ1 win-condition check complete.")


if __name__ == "__main__":
    main()
