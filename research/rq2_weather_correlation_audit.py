"""RQ2 addendum -- test-selection audit of the sales/temperature correlation.

RQ2_report.md cites "temperature correlates only 0.02 with raw sales" as
supporting evidence that weather earns little accuracy. That figure is the
row-level (item x store x day, n=394,560) Pearson r between raw `sales` and raw
`temperature` -- not `temp_anomaly`, and not the numbers actually printed by
`02_EDA_updated.ipynb` cell 26 (aggregated r=+0.1716; item-level range
-0.248..+0.307). It also isn't robust to test choice: `sales` at row level is a
right-skewed, zero-inflated count (see RQ1's `zero_rate` column), which is
exactly the profile `ml-datasci/selecting-statistical-test` flags as needing a
rank-based measure rather than Pearson. This script re-runs the same comparison
with Spearman alongside Pearson, at all three granularities the EDA notebook
touches, with confidence intervals per `ml-datasci/reporting-effect-sizes`
(bare correlation point estimates are not an acceptable report).
"""
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
DATA_DIR = "../data"


def spearman_ci(rho, n, alpha=0.05):
    """Bonett & Wright (2000) SE for Spearman's rho, Fisher-z CI."""
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(rho)
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def pearson_ci(r, n, alpha=0.05):
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(r)
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def main():
    sales = pd.read_csv(f"{DATA_DIR}/sales_data_preprocessed.csv", parse_dates=["date"])
    weather = pd.read_csv(f"{DATA_DIR}/weather_preprocessed.csv", parse_dates=["date"])
    sw = sales.merge(weather[["date", "province", "temperature", "temp_anomaly"]],
                      on=["date", "province"], how="left")
    assert sw["temperature"].isna().sum() == 0, "weather join failed"

    rows = []

    # Row level: the exact granularity behind the report's "0.02" figure.
    for col in ["temperature", "temp_anomaly"]:
        r, _ = stats.pearsonr(sw["sales"], sw[col])
        rho, p = stats.spearmanr(sw["sales"], sw[col])
        r_lo, r_hi = pearson_ci(r, len(sw))
        rho_lo, rho_hi = spearman_ci(rho, len(sw))
        rows.append({"granularity": "row-level (item x store x day)", "variable": col,
                     "n": len(sw), "pearson_r": r, "pearson_ci_lo": r_lo, "pearson_ci_hi": r_hi,
                     "spearman_rho": rho, "spearman_p": p,
                     "spearman_ci_lo": rho_lo, "spearman_ci_hi": rho_hi})

    # Aggregated level: matches 02_EDA_updated.ipynb cell 26's headline number.
    agg = sw.groupby(["date", "province"]).agg(
        sales=("sales", "sum"), temp_anomaly=("temp_anomaly", "first")).reset_index()
    r, _ = stats.pearsonr(agg["sales"], agg["temp_anomaly"])
    rho, p = stats.spearmanr(agg["sales"], agg["temp_anomaly"])
    r_lo, r_hi = pearson_ci(r, len(agg))
    rho_lo, rho_hi = spearman_ci(rho, len(agg))
    rows.append({"granularity": "aggregated (date x province sum)", "variable": "temp_anomaly",
                 "n": len(agg), "pearson_r": r, "pearson_ci_lo": r_lo, "pearson_ci_hi": r_hi,
                 "spearman_rho": rho, "spearman_p": p,
                 "spearman_ci_lo": rho_lo, "spearman_ci_hi": rho_hi})

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))

    # Item level: sign-agreement check between Pearson and Spearman per item,
    # plus a bootstrap CI on the median (n=30 items is too small for the
    # closed-form Fisher-z CI to be trustworthy -- same call the codebase
    # already makes for item-level Spearman in 06_proof_EXECUTED_example.ipynb).
    item_pearson = sw.groupby("item_name").apply(lambda g: g["sales"].corr(g["temp_anomaly"]))
    item_spearman = sw.groupby("item_name").apply(
        lambda g: stats.spearmanr(g["sales"], g["temp_anomaly"]).statistic)
    sign_flips = int((np.sign(item_pearson) != np.sign(item_spearman)).sum())

    rng = np.random.default_rng(2025)
    vals = item_spearman.values
    boot_medians = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(5000)]
    med_lo, med_hi = np.percentile(boot_medians, [2.5, 97.5])

    item_summary = pd.DataFrame([{
        "n_items": len(vals),
        "pearson_range_lo": item_pearson.min(), "pearson_range_hi": item_pearson.max(),
        "spearman_range_lo": item_spearman.min(), "spearman_range_hi": item_spearman.max(),
        "sign_flips_pearson_vs_spearman": sign_flips,
        "spearman_median": np.median(vals),
        "spearman_median_bootstrap_ci_lo": med_lo, "spearman_median_bootstrap_ci_hi": med_hi,
    }])
    print("\n" + item_summary.round(4).to_string(index=False))

    result.to_csv(f"{RESULTS_DIR}/rq2_weather_correlation.csv", index=False)
    item_summary.to_csv(f"{RESULTS_DIR}/rq2_weather_correlation_item_level.csv", index=False)

    print(
        "\nRow-level sales-vs-temperature SIGN-FLIPS between Pearson "
        f"({result.iloc[0]['pearson_r']:+.4f}) and Spearman "
        f"({result.iloc[0]['spearman_rho']:+.4f}) -- the '0.02' figure this "
        "audit was triggered by is the least robust of the three granularities."
    )


if __name__ == "__main__":
    main()
