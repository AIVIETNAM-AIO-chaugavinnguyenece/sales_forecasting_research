"""Shared utilities for the RQ1-RQ4 research evaluation scripts.

Conventions ported from notebooks/04_modelling_update.ipynb and
notebooks/06_proof_EXECUTED_example.ipynb so all four RQ scripts use an
identical train/fit/val/test boundary and an identical local-series sample,
which is required for their results to be comparable to each other.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 2025
N_LOCAL_SERIES = 24
TARGET = "sales"
DROP_COLS = ["date", "is_test", "store_item", "promo_id", "store_name", "item_name"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "..", "data", "feature_engineered_data_69_features.parquet")

# Regressor groups used by RQ2's leave-one-group-out ablation. Partitions the
# 62 model features (see models/model_metadata.json) by domain; anything not
# listed here (identifiers, price, lag/rolling features) is kept in every variant.
WEATHER_COLS = [
    "temperature", "humidity", "season", "temp_norm", "temp_anomaly",
    "is_humid", "temp_category", "humidity_level", "heat_excess", "cold_excess",
]
CALENDAR_COLS = [
    "is_public_holiday", "holiday_name", "is_school_holiday", "days_to_holiday",
    "days_since_holiday", "year", "month", "day", "day_of_week", "is_weekend",
    "quarter", "week_of_year", "day_of_year", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "days_to_christmas", "is_pre_holiday",
]
PROMO_COLS = [
    "is_promotion", "discount_pct", "promo_type", "duration_days",
    "log_price_ratio", "is_deep_discount", "rival_discount_pct",
]

# Held-out generator ground truth (never given to any model), from
# notebooks/06_proof_EXECUTED_example.ipynb cell 23156b49.
ITEM_TRUTH = pd.DataFrame([
    (1, 0.000, -0.55), (2, -0.004, -0.70), (3, 0.000, -0.45), (4, -0.008, -0.60),
    (5, -0.006, -0.80), (6, 0.000, -0.50), (7, 0.002, -0.40), (8, 0.000, -0.95),
    (9, 0.010, -0.90), (10, 0.045, -1.35), (11, -0.006, -0.75), (12, -0.025, -1.00),
    (13, 0.035, -1.45), (14, 0.020, -1.15), (15, 0.042, -1.10), (16, -0.024, -1.20),
    (17, -0.020, -0.85), (18, 0.050, -1.30), (19, 0.015, -1.50), (20, -0.005, -1.25),
    (21, -0.018, -1.40), (22, 0.000, -1.15), (23, 0.004, -1.10), (24, 0.000, -0.90),
    (25, 0.000, -1.05), (26, 0.000, -0.75), (27, 0.000, -0.80), (28, 0.060, -1.10),
    (29, 0.045, -0.95), (30, -0.045, -0.70),
], columns=["item_id", "temp_effect", "elasticity"])
ITEM_TRUTH["abs_temp_effect"] = ITEM_TRUTH["temp_effect"].abs()
ITEM_TRUTH["abs_elasticity"] = ITEM_TRUTH["elasticity"].abs()


def load_features(path=DATA_PATH):
    return pd.read_parquet(path)


def feature_cols(df, drop_extra=None):
    drop = set(DROP_COLS + [TARGET] + (drop_extra or []))
    return [c for c in df.columns if c not in drop]


def time_split(df):
    """Leak-free split: last 28 days of training held out as a validation set
    for early stopping, so early stopping never sees the test set."""
    train_mask = ~df["is_test"]
    cutoff = df.loc[~train_mask, "date"].min()
    val_start = cutoff - pd.Timedelta(days=28)
    fit_mask = train_mask & (df["date"] < val_start)
    val_mask = train_mask & (df["date"] >= val_start)
    return train_mask, fit_mask, val_mask, cutoff


def select_local_series(df, train_mask, n=N_LOCAL_SERIES, seed=SEED):
    """Series sampled evenly across the volume range (not randomly), so low-
    and high-volume series are both represented."""
    series_vol = df[train_mask].groupby("store_item")["sales"].mean().sort_values()
    if n is None:
        return series_vol.index.tolist()
    idx = np.linspace(0, len(series_vol) - 1, n).astype(int)
    return series_vol.index[idx].tolist()


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def mase(y_true, y_pred, y_train, season=1):
    y_true, y_pred, y_train = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_train)
    naive = np.mean(np.abs(np.diff(y_train, n=season)))
    if not np.isfinite(naive) or naive <= 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / naive


def score(y_true, y_pred, y_train=None, label=""):
    out = {
        "model": label,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "WAPE": wape(y_true, y_pred),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train)
    return out


def diebold_mariano(e1, e2):
    """Diebold-Mariano test on squared-error loss differentials."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    var = np.var(d, ddof=0) / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


LGBM_PARAMS = dict(
    objective="regression", metric="rmse", boosting_type="gbdt", num_leaves=63,
    learning_rate=0.05, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    min_child_samples=40, n_estimators=2000, random_state=SEED, n_jobs=-1, verbose=-1,
)

XGB_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse", n_estimators=2000, learning_rate=0.05,
    max_depth=8, min_child_weight=10, subsample=0.85, colsample_bytree=0.85,
    tree_method="hist", enable_categorical=True, early_stopping_rounds=100,
    random_state=SEED, n_jobs=-1, verbosity=0,
)


def fit_lgbm(X_fit, y_fit, X_val, y_val, cat_features):
    import lightgbm as lgbm
    model = lgbm.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="rmse",
        callbacks=[lgbm.early_stopping(100, verbose=False), lgbm.log_evaluation(0)],
        categorical_feature=cat_features,
    )
    return model


def fit_xgb(X_fit, y_fit, X_val, y_val):
    import xgboost as xgb
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_sarimax(panel, series_ids, exog_cols=None, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
    """Fit one SARIMAX per series and forecast the test window.

    exog_cols, if given, must be column names present in panel: passing the
    exog as a named DataFrame slice (not .values) is required so that
    fit.param_names carries the regressor names for later coefficient lookup.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rows, failures = [], 0
    for sid in series_ids:
        g = panel[panel["store_item"] == sid]
        # reset_index is required, not cosmetic: statsmodels' date-index
        # inference is unreliable on the non-contiguous index left behind by
        # filtering a multi-series panel, and silently raises inside
        # .forecast() when exog is a named DataFrame -- caught below as a
        # "failure" and masked by the naive fallback, so this must be fixed
        # here rather than left for the except clause to paper over.
        tr = g[~g["is_test"]].reset_index(drop=True)
        te = g[g["is_test"]].reset_index(drop=True)
        if len(tr) < 60 or len(te) == 0:
            continue
        exog_tr = tr[exog_cols] if exog_cols else None
        exog_te = te[exog_cols] if exog_cols else None
        try:
            model = SARIMAX(
                tr["sales"].values, exog=exog_tr, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=100)
            fc = fit.forecast(steps=len(te), exog=exog_te)
        except Exception as e:
            failures += 1
            print(f"    [fallback] {sid}: {type(e).__name__}: {e}")
            fc = np.repeat(tr["sales"].tail(7).mean(), len(te))
        rows.append(pd.DataFrame({
            "store_item": sid,
            "date": te["date"].values,
            "actual": te["sales"].values,
            "pred": np.maximum(np.asarray(fc, dtype=float), 0),
        }))
    print(f"  SARIMAX: {failures} fallbacks / {len(series_ids)} series")
    return pd.concat(rows, ignore_index=True)
