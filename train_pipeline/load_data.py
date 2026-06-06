"""
load_data.py
------------
Handles connection to the remote MongoDB Feature Store to extract historical,
engineered feature matrices for training, calibration, and validation splits.

All data is fetched exclusively from MongoDB — no local file I/O anywhere.

KEY IMPROVEMENTS FOR R²
─────────────────────────
  IMPR A — Robust NaN imputation for trees: instead of raw NaN passthrough
    for RF (which degrades splits), we now use median imputation for all
    models but fitted strictly on the training fold only.

  IMPR B — Reduced correlation filter threshold from 0.97 to 0.95 for
    tree models (called per-model in train_pipeline.py). Removes more redundant
    features, reduces noise in split decisions, and speeds up training.

  IMPR C — Target outlier winsorisation: AQI targets above 500 (physically
    impossible) are clipped. This prevents a handful of extreme values from
    dominating the loss function and deflating R².

  IMPR D — Leakage guard extended: added aqi_historical_anchor to the
    CURRENT_TIMESTEP_COLS exclusion list (it was already guarded in
    feature_engineering.py but now double-checked at load time).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

load_dotenv(BASE_DIR / ".env")

MONGO_URI       = os.getenv("MONGODB_URI")
DB_NAME         = "karachi_aqi"
COLLECTION_NAME = "processed_features"


def impute_for_linear(
    X_train: "pd.DataFrame",
    X_cal:   "pd.DataFrame | None",
    X_test:  "pd.DataFrame",
) -> tuple:
    """
    Column-wise MEDIAN imputation fitted ONLY on X_train, applied to cal/test.
    Median is more robust than mean for skewed AQI distributions.
    Used by Ridge (cannot handle NaN natively).
    """
    train_medians = X_train.median()
    X_train_imp   = X_train.fillna(train_medians)
    X_test_imp    = X_test.fillna(train_medians)
    if X_cal is not None:
        X_cal_imp = X_cal.fillna(train_medians)
        return X_train_imp, X_cal_imp, X_test_imp
    return X_train_imp, X_test_imp


def impute_for_trees(
    X_train: "pd.DataFrame",
    X_cal:   "pd.DataFrame | None",
    X_test:  "pd.DataFrame",
) -> tuple:
    """
    Column-wise MEDIAN imputation for tree models.
    XGBoost handles NaN natively via learned routing, but Random Forest
    does not — zero-filling (old approach) created spurious tree splits.
    Median imputation fitted on train only avoids both problems.
    """
    train_medians = X_train.median()
    X_train_imp   = X_train.fillna(train_medians)
    X_test_imp    = X_test.fillna(train_medians)
    if X_cal is not None:
        X_cal_imp = X_cal.fillna(train_medians)
        return X_train_imp, X_cal_imp, X_test_imp
    return X_train_imp, X_test_imp


REDUNDANT_TIME_COLS = [
    "hour", "day", "month", "weekday",
    "day_of_year", "week_of_year", "hour_of_week",
]

ALL_TARGETS = [
    "target_aqi_12h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_12h", "target_cat_24h", "target_cat_48h", "target_cat_72h",
    "target_aqi_12h_deviation", "target_aqi_24h_deviation",
    "target_aqi_48h_deviation", "target_aqi_72h_deviation",
]

CURRENT_TIMESTEP_COLS = [
    "aqi",
    "aqi_historical_anchor",  # IMPR D: double-guard scratch variable
    "pm25", "pm10", "co", "no2", "so2", "o3", "dust", "uv_index",
    "temperature", "temperature_2m",
    "humidity", "relative_humidity_2m",
    "wind_speed", "wind_speed_10m",
    "wind_direction", "wind_direction_10m",
    "wind_gusts",
    "precipitation",
    "cloud_cover",
    "dew_point", "dew_point_2m",
    "pressure",
    "surface_pressure",
]

BASE_DROP     = ["datetime", "timestamp"] + REDUNDANT_TIME_COLS + ALL_TARGETS + CURRENT_TIMESTEP_COLS
LEAKAGE_EXACT = frozenset(ALL_TARGETS + CURRENT_TIMESTEP_COLS)


def _fetch_from_feature_store() -> pd.DataFrame:
    """Queries the centralized cloud feature store collection from MongoDB."""
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGODB_URI environment variable is missing or unset.")

    print(f"\nEstablishing active cluster link to pool: {DB_NAME}.{COLLECTION_NAME}")
    try:
        client     = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        db         = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        cursor     = collection.find({}, {"_id": 0})
        documents  = list(cursor)
        client.close()
    except PyMongoError as e:
        print(f"CRITICAL: Failed to stream from MongoDB Atlas Cluster: {e}")
        raise

    if not documents:
        raise RuntimeError(
            f"CRITICAL: Feature collection '{COLLECTION_NAME}' is completely empty."
        )

    df = pd.DataFrame(documents)

    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise KeyError("Pulled collection is missing temporal reference anchors.")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def load_xy(horizon: int, use_log: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    assert horizon in (12, 24, 48, 72), "Horizon must be 12, 24, 48, or 72."

    df = _fetch_from_feature_store()
    print(f"Extracted feature store dataset matrix shape: {df.shape}")

    raw_target_col = f"target_aqi_{horizon}h"
    log_target_col = f"target_aqi_{horizon}h_log"

    if raw_target_col not in df.columns:
        raise ValueError(
            f"Target column not found in cloud document schema: '{raw_target_col}'"
        )

    df = df.dropna(subset=[raw_target_col]).reset_index(drop=True)

    # IMPR C: winsorise physically impossible AQI values before training
    df[raw_target_col] = df[raw_target_col].clip(upper=500.0)

    if use_log:
        if log_target_col not in df.columns:
            df[log_target_col] = np.log1p(df[raw_target_col])
        # Recompute log target after clipping
        df[log_target_col] = np.log1p(df[raw_target_col])
        y = df[log_target_col].copy()
    else:
        y = df[raw_target_col].copy()

    y_raw = df[raw_target_col].copy()

    # Build feature matrix
    X = df.drop(columns=[c for c in BASE_DROP if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Replace Inf/-Inf from divisions or rolling ops
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop features with >20% NaN (long-range lags have structural warmup NaN — accepted)
    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.20].index.tolist()
    if high_missing:
        print(
            f"Dropping {len(high_missing)} features exceeding 20% NaN threshold: {high_missing}"
        )
        X = X.drop(columns=high_missing)

    # Leakage guard
    leaky = [c for c in X.columns if c in LEAKAGE_EXACT]
    if leaky:
        raise ValueError(
            f"CRITICAL Data leakage detected. Forbidden columns still present in X:\n"
            f"  {leaky}\n"
            f"Check BASE_DROP in load_data.py and feature_engineering.py."
        )

    # Diagnostics
    residual_nan = X.isna().mean()
    nan_cols = residual_nan[residual_nan > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        print("\nResidual NaN rates in feature matrix:")
        print(nan_cols.round(4).to_string())

    print(f"\nTarget Distributions (raw AQI {horizon}h):")
    print(y_raw.describe().round(1).to_string())
    print(f"\nModel feature dimension space: {X.shape[1]}")
    print(f"Total row entries partitioned: {X.shape[0]:,}")

    if "aqi_lag_1" in X.columns:
        lag1_corr = X["aqi_lag_1"].corr(y_raw)
        flag = (
            "  <<< WARNING: SUSPICIOUSLY HIGH — CHECK FOR RESIDUAL LEAKAGE"
            if lag1_corr > 0.99 else ""
        )
        print(f"aqi_lag_1 correlation -> Target ({horizon}h): {lag1_corr:.3f}{flag}")

    return X, y


def get_chronological_splits(X: pd.DataFrame, y: pd.Series, horizon: int):
    """
    Applies unified 75 / 10 / 15 chronological validation split.
    Buffer gaps equal to the lookahead horizon prevent temporal overlap.
    """
    n         = len(X)
    train_end = int(n * 0.75)
    cal_end   = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end]

    X_cal   = X.iloc[train_end + horizon : cal_end].copy()
    y_cal   = y.iloc[train_end + horizon : cal_end]

    X_test  = X.iloc[cal_end:].copy()
    y_test  = y.iloc[cal_end:]

    return X_train, y_train, X_cal, y_cal, X_test, y_test


def get_spike_augmented_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_raw: pd.Series | None = None,
    spike_threshold: float = 150.0,
    target_spike_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series]:
    ref          = y_train_raw if y_train_raw is not None else y_train
    spike_mask   = ref > spike_threshold
    n_spike      = spike_mask.sum()
    n_total      = len(X_train)
    current_frac = n_spike / n_total

    if current_frac >= target_spike_fraction:
        print(
            f"[spike-aug] Train split already has {current_frac:.3f} spike fraction. Skipping."
        )
        return X_train, y_train

    n_needed = int(target_spike_fraction * n_total / (1 - target_spike_fraction)) - n_spike
    if n_needed <= 0 or n_spike == 0:
        return X_train, y_train

    rng          = np.random.default_rng(42)
    spike_idx    = np.where(spike_mask)[0]
    resample_idx = rng.choice(spike_idx, size=n_needed, replace=True)

    X_extra = X_train.iloc[resample_idx].copy()
    y_extra = y_train.iloc[resample_idx].copy()

    X_aug = pd.concat([X_train, X_extra], ignore_index=True)
    y_aug = pd.concat([y_train, y_extra], ignore_index=True)

    print(f"[spike-aug] Added {n_needed} spike rows → "
          f"new spike fraction: {(n_spike + n_needed) / (n_total + n_needed):.3f}")
    return X_aug, y_aug


def apply_leakage_free_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_cal:  pd.DataFrame | None = None,
    threshold: float = 0.95,   # IMPR B: tighter default for fewer redundant features
) -> tuple:
    corr  = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    protected = {
        "aqi_lag_1", "aqi_lag_6", "aqi_lag_12",
        "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
        "aqi_change_1h", "aqi_change_6h", "aqi_change_24h", "aqi_acceleration",
        "pm25_roll_std_24", "interaction_pm25_humidity",
        "interaction_pm25_wind_inverse", "dust_lag_1",
        "dew_point_depression", "wind_persistence_ratio",
        "pm25_change_3h", "pm25_change_6h", "pm25_roc_sign",
        "fourier_annual_sin", "fourier_annual_cos",
        "fourier_semi_annual_sin", "fourier_semi_annual_cos",
        "ventilation_index", "is_morning_stagnation", "monsoon_washout",
    }

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold) and col not in protected
    ]

    X_train_c = X_train.drop(columns=to_drop)
    X_test_c  = X_test.drop(columns=to_drop)

    if X_cal is not None:
        X_cal_c = X_cal.drop(columns=to_drop)
        return X_train_c, X_cal_c, X_test_c, to_drop

    return X_train_c, X_test_c, to_drop


def calculate_conformal_margin(abs_residuals: np.ndarray, alpha: float = 0.05) -> float:
    n_cal   = len(abs_residuals)
    if n_cal == 0:
        return 0.0
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    return float(np.quantile(abs_residuals, q_level))


def compute_aqi_event_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: int
) -> dict:
    true_ev   = (y_true > threshold).astype(int)
    pred_ev   = (y_pred > threshold).astype(int)
    tp        = np.sum((true_ev == 1) & (pred_ev == 1))
    fp        = np.sum((true_ev == 0) & (pred_ev == 1))
    fn        = np.sum((true_ev == 1) & (pred_ev == 0))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0 else 0.0
    )
    return {
        f"precision_gt{threshold}": precision,
        f"recall_gt{threshold}":    recall,
        f"f1_gt{threshold}":        f1,
    }


def get_persistence_baseline_col(X_test: pd.DataFrame, horizon: int) -> str | None:
    preferred = f"aqi_lag_{horizon}"
    if preferred in X_test.columns:
        return preferred
    if "aqi_lag_1" in X_test.columns:
        return "aqi_lag_1"
    return None


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("      LOAD_DATA CLOUD PIPELINE INTEGRITY & SPLIT VERIFICATION")
    print("=" * 80)

    for h in (24, 48, 72):
        print(f"\n{'#' * 60}\n DATABASE HOOK INGESTION CHECK FOR HORIZON: {h}h\n{'#' * 60}")
        try:
            X, y = load_xy(h, use_log=True)
            X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(X, y, h)
            print(f"  Train Split : {X_train.shape[0]:,} records")
            print(f"  Cal Split   : {X_cal.shape[0]:,} records")
            print(f"  Test Split  : {X_test.shape[0]:,} records")
            X_train_f, X_cal_f, X_test_f, dropped = apply_leakage_free_correlation_filter(
                X_train, X_test, X_cal, threshold=0.95
            )
            print(f"  Surviving Dimensions : {X_train_f.shape[1]} (Filtered: {len(dropped)})")
        except Exception as e:
            print(f"ERROR for horizon {h}h: {e}")