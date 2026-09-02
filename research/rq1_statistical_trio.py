"""RQ1 addendum -- statistical trio on the LightGBM/XGBoost/SARIMA accuracy deltas.

Diebold-Mariano (see rq1_model_comparison.py) is the correct paired test here:
all three models are scored on the identical 24-series / n=2,208 test window, and
DM operates directly on the paired squared-error loss differential -- the
point-forecast analogue of a paired t-test on per-instance error, which is what
`comparing-models-fairly` recommends for "same test instances, regression models".
This script adds the other two legs of the trio: an effect-size CI on the MSE
differential, and Holm-Bonferroni correction across the 3 pairwise comparisons
(required once m >= 3 comparisons are run on the same experiment).

The per-row prediction frame from rq1_model_comparison.py's `comparison` variable
isn't persisted to disk, so a resampling bootstrap over rows/series isn't
available without re-fitting all three models. Instead the CI is recovered
analytically from the same moments the DM test itself uses: DM's
d = e_a^2 - e_b^2 has d.mean() = MSE_a - MSE_b (available from the RMSE column in
rq1_overall.csv), and Var(d)/n is backed out from the stored
dm_stat = d.mean() / sqrt(Var(d)/n). This is the exact companion CI to a
Wald-style DM statistic, not an approximation -- it shares the DM test's own
asymptotic-normality assumption rather than adding a separate one.
"""
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"


def analytic_mse_ci(rmse_a, rmse_b, dm_stat, alpha=0.05):
    d_bar = rmse_a**2 - rmse_b**2
    se = abs(d_bar / dm_stat) if dm_stat != 0 else np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    return d_bar, d_bar - z * se, d_bar + z * se


def holm_bonferroni(pvalues, alpha=0.05):
    """Step-down Holm-Bonferroni. Returns (reject, adjusted_alpha) aligned to input order."""
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
            break  # step-down: once one comparison fails to reject, so do all later ones
    return reject, adj_alpha


def main():
    overall = pd.read_csv(f"{RESULTS_DIR}/rq1_overall.csv").set_index("model")
    dm = pd.read_csv(f"{RESULTS_DIR}/rq1_dm_tests.csv")

    rows = []
    for _, r in dm.iterrows():
        a, b, stat, p = r["model_a"], r["model_b"], r["dm_stat"], r["p_value"]
        d_bar, lo, hi = analytic_mse_ci(overall.loc[a, "RMSE"], overall.loc[b, "RMSE"], stat)
        rows.append({
            "model_a": a, "model_b": b, "dm_stat": stat, "p_value": p,
            "mse_delta_a_minus_b": d_bar, "mse_delta_ci_lo": lo, "mse_delta_ci_hi": hi,
            "wape_delta_pp_a_minus_b": overall.loc[a, "WAPE"] - overall.loc[b, "WAPE"],
        })
    trio = pd.DataFrame(rows)
    reject, adj_alpha = holm_bonferroni(trio["p_value"].tolist())
    trio["holm_adjusted_alpha"] = adj_alpha
    trio["significant_after_holm"] = reject

    print(trio.round(4).to_string(index=False))
    trio.to_csv(f"{RESULTS_DIR}/rq1_statistical_trio.csv", index=False)
    print(
        "\nAll 3 pairwise DM differences survive Holm-Bonferroni correction "
        f"(m=3, alpha=0.05): {bool(trio['significant_after_holm'].all())}"
    )


if __name__ == "__main__":
    main()
