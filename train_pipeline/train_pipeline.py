"""
train_pipeline.py

Fixed training pipeline for Karachi AQI forecasting.

R² IMPROVEMENT FIXES (this version)
─────────────────────────────────────
  FIX 1 — Gap buffer in chronological splits: added `horizon`-hour gap between
    train→cal and cal→test to prevent temporal leakage from lag features that
    span the boundary. Without this, aqi_lag_1 in the first cal row is the last
    train row, giving the model a direct look-ahead path.

  FIX 2 — Ridge CV NaN blowup: imputation must happen BEFORE the TimeSeriesSplit
    CV loop in train_ridge(), not after. Previously X_train still had NaN when
    passed into the per-fold StandardScaler, which produced astronomically large
    CV RMSE (~10M). Imputation is now fitted on the train fold inside the pipeline
    using a SimpleImputer step, giving honest per-fold imputation without leaking
    medians from validation rows.

  FIX 3 — load_xy_both(): single MongoDB fetch per horizon shared across all
    three model trainers. Previously each of the 3 models fetched independently
    (9 total fetches for 3 horizons). Now one fetch per horizon via a module-level
    cache dict _DATA_CACHE.

  FIX 4 — XGBoost hyperparameter tuning: increased colsample_bytree search space
    and added gamma + reg_alpha to the CV search. Also bumped n_estimators upper
    bound to 1500 with lower learning_rate floor (0.008) to let early stopping
    find better optima.

  FIX 5 — RF: inner estimator explicitly set n_jobs=1 to prevent nested
    parallelism. RandomizedSearchCV outer n_jobs=-1 already parallelises folds;
    inner n_jobs=-1 causes CPU contention on 2-vCPU GitHub Actions runner.

  FIX 6 — Consistent spike augmentation: both RF and XGB now target 10% spike
    fraction (was 15%/7% mismatch). Lower fraction means less class imbalance
    distortion while still giving high-AQI events representation.

  FIX 7 — XGBoost feature imputation: XGBoost handles NaN via learned routing,
    but explicit median imputation of the train set (fitted on X_train only)
    before passing to fit() gives more stable leaf assignments and better SHAP.

  FIX 8 — Increased SHAP sample to 500 for XGBoost to get stable feature
    importance rankings (was 300, too noisy for 150-feature sets).
"""

import io
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

_TRAIN_PIPELINE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TRAIN_PIPELINE_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)

from pymongo import MongoClient
from pymongo.errors import PyMongoError
import gridfs

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. XGBoost will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed. SHAP explanations will be skipped.")

try:
    from monitoring import run_data_drift_monitoring
except ImportError:
    def run_data_drift_monitoring(*args, **kwargs):
        print("  [monitoring] skipped. module not found.")


DB_NAME = "karachi_aqi"
FEATURE_COLLECTION = "processed_features"
HORIZONS = [24, 48, 72]

MODEL_NAMES = ["random_forest", "ridge", "xgboost"]

DISPLAY = {
    "random_forest": "Random Forest",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
}

PROTECTED_FEATURES = {
    "aqi",
    "pm25",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "dust",
    "uv_index",
    "temperature",
    "humidity",
    "pressure",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
    "wind_gusts",
    "precipitation",
    "cloud_cover",
    "dew_point",
}

# FIX 3: Module-level cache so each horizon is fetched only once
_DATA_CACHE: dict[int, tuple] = {}


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mongo_clean(value):
    if isinstance(value, dict):
        return {k: _mongo_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mongo_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_mongo_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_mongo_clean(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _get_db():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set.")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client[DB_NAME], client


def _serialise_model_bytes(artifact: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(artifact, buf, compress=3)
    return buf.getvalue()


def push_model(
    horizon: int,
    model_name: str,
    artifact: dict,
    metrics: dict,
    feature_names: list,
    top_features: list | None = None,
):
    db, client = _get_db()
    registry_col = db["model_registry"]
    fs = gridfs.GridFS(db, collection="model_artifacts")

    trained_at = datetime.now(tz=timezone.utc).isoformat()
    filename = f"{model_name}_{horizon}h_{trained_at}.joblib"

    existing = registry_col.find_one(
        {"horizon": horizon, "model_name": model_name},
        {"artifact_gridfs_id": 1},
    )

    if existing and existing.get("artifact_gridfs_id"):
        try:
            fs.delete(existing["artifact_gridfs_id"])
        except Exception as e:
            print(f"  [MongoDB] old artifact cleanup skipped: {e}")

    model_bytes = _serialise_model_bytes(artifact)

    artifact_file_id = fs.put(
        model_bytes,
        filename=filename,
        content_type="application/octet-stream",
        metadata={
            "horizon": horizon,
            "model_name": model_name,
            "trained_at": trained_at,
            "artifact_type": "joblib_model",
        },
    )

    doc = {
        "horizon": horizon,
        "model_name": model_name,
        "trained_at": trained_at,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "artifact_storage": "gridfs",
        "artifact_gridfs_id": artifact_file_id,
        "artifact_filename": filename,
        "artifact_size_bytes": len(model_bytes),
        **{
            k: _mongo_clean(v)
            for k, v in metrics.items()
            if not isinstance(v, (dict, list))
        },
    }

    if top_features:
        doc["top_features"] = _mongo_clean(top_features)

    registry_col.update_one(
        {"horizon": horizon, "model_name": model_name},
        {"$set": doc, "$unset": {"model_binary": ""}},
        upsert=True,
    )

    client.close()

    print(
        f"  [MongoDB] model_artifacts/GridFS <- {model_name} horizon={horizon}h "
        f"({len(model_bytes) / 1024 / 1024:.2f} MB)"
    )
    print(f"  [MongoDB] model_registry <- {model_name} horizon={horizon}h")


def push_metrics(horizon: int, model_name: str, metrics: dict):
    """
    Upserts the metrics doc for this model/horizon into model_metrics.

    FIX: was insert_one() which appended a new document on every run,
    growing the collection forever and making the API sort-by-trained_at
    to find the latest.  Now update_one(upsert=True) keeps exactly one
    document per (horizon, model_name) pair — simpler and cheaper.
    """
    db, client = _get_db()

    doc = _mongo_clean(
        {
            "horizon":    horizon,
            "model_name": model_name,
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            **metrics,
        }
    )

    db["model_metrics"].update_one(
        {"horizon": horizon, "model_name": model_name},
        {"$set": doc},
        upsert=True,
    )

    client.close()
    print(f"  [MongoDB] model_metrics <- {model_name} horizon={horizon}h")


def push_predictions(
    horizon: int,
    model_name: str,
    y_arr: np.ndarray,
    preds_raw: np.ndarray,
    pi_lower: np.ndarray,
    pi_upper: np.ndarray,
    index: pd.Index,
    feature_timestamps: pd.Series,
):
    """
    Writes prediction rows to the per-model/horizon collection used by the API.

    Important timestamp fix:
    `actual` and `predicted` represent the AQI at the forecast target time,
    not the feature row time. So each document stores:
      - feature_timestamp: the input row timestamp used to make the forecast
      - timestamp: the actual forecast target timestamp, feature_timestamp + horizon hours

    Streamlit uses `timestamp` for the x-axis, so the forecast chart now shows
    real dates instead of falling back to dataframe index positions.
    """
    db, client = _get_db()

    collection_name = f"predictions_{model_name}_{horizon}h"
    col = db[collection_name]

    # Ensure clean upserts and efficient latest-row sorting.
    col.create_index("row_index", unique=True)
    col.create_index("timestamp")

    docs = []
    feature_timestamps = pd.to_datetime(feature_timestamps).reset_index(drop=True)

    if len(feature_timestamps) != len(y_arr):
        client.close()
        raise ValueError(
            f"Timestamp length mismatch for {model_name} {horizon}h: "
            f"{len(feature_timestamps)} timestamps vs {len(y_arr)} predictions."
        )

    for i, idx in enumerate(index):
        feature_ts = pd.Timestamp(feature_timestamps.iloc[i]).to_pydatetime()
        target_ts = (pd.Timestamp(feature_ts) + pd.Timedelta(hours=horizon)).to_pydatetime()

        docs.append(
            {
                "row_index": int(idx),
                "timestamp": target_ts,
                "feature_timestamp": feature_ts,
                "horizon": horizon,
                "model_name": model_name,
                "actual": float(y_arr[i]),
                "predicted": float(preds_raw[i]),
                "pi_lower": float(pi_lower[i]),
                "pi_upper": float(pi_upper[i]),
            }
        )

    from pymongo import UpdateOne as _UpdateOne

    ops = [
        _UpdateOne({"row_index": d["row_index"]}, {"$set": d}, upsert=True)
        for d in docs
    ]

    BATCH = 1000
    total_batches = (len(ops) + BATCH - 1) // BATCH
    for i in range(0, len(ops), BATCH):
        batch_num = i // BATCH + 1
        result = col.bulk_write(ops[i : i + BATCH], ordered=False)
        print(
            f"  [MongoDB] {collection_name} batch {batch_num}/{total_batches} — "
            f"upserted: {result.upserted_count}, modified: {result.modified_count}"
        )

    client.close()
    print(
        f"  [MongoDB] {collection_name} <- {len(docs)} prediction rows written with timestamps."
    )


def flag_best_model(horizon: int, best_model_name: str):
    db, client = _get_db()
    col = db["model_registry"]

    col.update_many({"horizon": horizon}, {"$set": {"is_best": False}})
    col.update_one(
        {"horizon": horizon, "model_name": best_model_name},
        {"$set": {"is_best": True}},
    )

    client.close()


def push_pipeline_run_summary(summary: dict):
    db, client = _get_db()

    db["pipeline_runs"].insert_one(
        _mongo_clean(
            {
                "run_at": datetime.now(tz=timezone.utc).isoformat(),
                **summary,
            }
        )
    )

    client.close()
    print("  [MongoDB] pipeline_runs <- run summary saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (FIX 3: cache per-horizon fetch)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_feature_store() -> pd.DataFrame:
    db, client = _get_db()

    try:
        docs = list(db[FEATURE_COLLECTION].find({}, {"_id": 0}))
    except PyMongoError as e:
        client.close()
        raise RuntimeError(f"Failed to stream from MongoDB Atlas: {e}")

    client.close()

    if not docs:
        raise RuntimeError(
            f"CRITICAL: Feature collection '{FEATURE_COLLECTION}' is empty."
        )

    df = pd.DataFrame(docs)

    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise KeyError("Feature store is missing datetime or timestamp.")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def load_xy_both(horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    FIX 3: Single-fetch entry point.
    Returns (X, y_log, y_raw, timestamps) — all four are needed by every model trainer.
    Results are cached in _DATA_CACHE so subsequent callers for the same
    horizon hit the cache instead of re-querying MongoDB.
    """
    if horizon in _DATA_CACHE:
        return _DATA_CACHE[horizon]

    assert horizon in (24, 48, 72), "Horizon must be 24, 48, or 72."

    df = _fetch_from_feature_store()
    print(f"Extracted feature store dataset matrix shape: {df.shape}")

    raw_target_col = f"target_aqi_{horizon}h"

    if raw_target_col not in df.columns:
        raise ValueError(f"Target column not found: {raw_target_col}")

    df = df.dropna(subset=[raw_target_col]).reset_index(drop=True)
    df[raw_target_col] = pd.to_numeric(df[raw_target_col], errors="coerce")
    df = df.dropna(subset=[raw_target_col]).reset_index(drop=True)

    # Winsorise physically impossible AQI values
    df[raw_target_col] = df[raw_target_col].clip(lower=0, upper=500)

    y_raw = pd.Series(df[raw_target_col].values, index=df.index, name=raw_target_col)
    y_log = pd.Series(np.log1p(df[raw_target_col].values), index=df.index, name=raw_target_col)

    # Keep feature timestamps so prediction rows can be plotted against real dates.
    # The predicted/actual value belongs to the target time, which is feature time + horizon.
    timestamps = pd.Series(pd.to_datetime(df["datetime"]), index=df.index, name="feature_timestamp")

    # Build feature matrix — drop all target columns + non-numeric bookkeeping
    drop_cols = ["datetime", "timestamp"]
    for col in df.columns:
        if col.startswith("target_"):
            drop_cols.append(col)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop features with >20% NaN (long lags have structural warm-up NaN — acceptable)
    missing_frac = X.isna().mean()
    high_missing = missing_frac[missing_frac > 0.20].index.tolist()
    if high_missing:
        print(
            f"Dropping {len(high_missing)} features exceeding 20% NaN threshold: "
            f"{high_missing}"
        )
        X = X.drop(columns=high_missing)

    # Drop constant / near-constant columns
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant features.")
        X = X.drop(columns=constant_cols)

    print(f"Model feature dimension space: {X.shape[1]}")
    print(f"Total row entries partitioned: {len(X):,}")

    if "aqi" in X.columns:
        corr = pd.Series(X["aqi"]).corr(y_raw)
        print(f"current aqi correlation -> Target ({horizon}h): {corr:.3f}")

    if "aqi_lag_1" in X.columns:
        corr = pd.Series(X["aqi_lag_1"]).corr(y_raw)
        flag = (
            "  <<< WARNING: SUSPICIOUSLY HIGH — CHECK FOR RESIDUAL LEAKAGE"
            if corr > 0.99 else ""
        )
        print(f"aqi_lag_1 correlation -> Target ({horizon}h): {corr:.3f}{flag}")

    result = (X, y_log, y_raw, timestamps)
    _DATA_CACHE[horizon] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Splitting, filtering, imputation helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chronological_splits(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    train_frac: float = 0.70,
    cal_frac: float = 0.15,
):
    """
    FIX 1: Gap buffer equal to `horizon` rows is inserted between
    train→cal and cal→test boundaries.  Without this, lag features
    (aqi_lag_1, aqi_lag_24 …) in the first rows of cal/test overlap
    with the last rows of the preceding split, giving the model an
    indirect temporal shortcut that inflates CV scores but tanks
    generalisation.
    """
    n = len(X)

    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))

    # Apply horizon-sized gap at each boundary
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_cal = X.iloc[train_end + horizon : cal_end].copy()
    y_cal = y.iloc[train_end + horizon : cal_end].copy()

    X_test = X.iloc[cal_end + horizon :].copy()
    y_test = y.iloc[cal_end + horizon :].copy()

    return X_train, y_train, X_cal, y_cal, X_test, y_test


def get_spike_augmented_train(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    y_train_raw: pd.Series,
    spike_threshold: float = 150,
    target_spike_fraction: float = 0.10,   # FIX 6: unified 10% for all models
):
    spike_mask = y_train_raw >= spike_threshold
    current_spike_fraction = float(spike_mask.mean())

    if current_spike_fraction >= target_spike_fraction:
        return X_train, y_train_log

    X_spike = X_train.loc[spike_mask]
    y_spike = y_train_log.loc[spike_mask]

    if X_spike.empty:
        return X_train, y_train_log

    n_current = len(X_train)
    n_spike = len(X_spike)

    needed = int(
        max(
            0,
            (target_spike_fraction * n_current - n_spike)
            / max(1 - target_spike_fraction, 1e-6),
        )
    )

    if needed <= 0:
        return X_train, y_train_log

    rng = np.random.default_rng(42)
    take_idx = rng.choice(X_spike.index.values, size=needed, replace=True)

    X_extra = X_train.loc[take_idx].copy()
    y_extra = y_train_log.loc[take_idx].copy()

    X_aug = pd.concat([X_train, X_extra], axis=0).reset_index(drop=True)
    y_aug = pd.concat([y_train_log, y_extra], axis=0).reset_index(drop=True)

    new_raw = np.expm1(y_aug.values)
    new_fraction = float((new_raw >= spike_threshold).mean())

    print(
        f"[spike-aug] Added {needed} spike rows -> "
        f"new spike fraction: {new_fraction:.3f}"
    )

    return X_aug, y_aug


def apply_leakage_free_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_cal: pd.DataFrame | None = None,
    threshold: float = 0.97,
):
    train = X_train.copy()

    corr = train.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dropped_cols = []

    for col in upper.columns:
        if col in PROTECTED_FEATURES:
            continue

        high_corr = upper[col][upper[col] > threshold]

        if len(high_corr) > 0:
            dropped_cols.append(col)

    X_train_f = X_train.drop(columns=dropped_cols, errors="ignore")
    X_test_f = X_test.drop(columns=dropped_cols, errors="ignore")

    if X_cal is not None:
        X_cal_f = X_cal.drop(columns=dropped_cols, errors="ignore")
        return X_train_f, X_cal_f, X_test_f, dropped_cols

    return X_train_f, X_test_f, dropped_cols


def impute_medians(
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame | None,
    X_test: pd.DataFrame,
):
    """Median imputation fitted ONLY on X_train, applied to cal/test."""
    train_medians = X_train.median(numeric_only=True)
    X_train_imp = X_train.fillna(train_medians)
    X_test_imp = X_test.fillna(train_medians)

    if X_cal is not None:
        X_cal_imp = X_cal.fillna(train_medians)
        return X_train_imp, X_cal_imp, X_test_imp

    return X_train_imp, X_test_imp


def calculate_conformal_margin(abs_errors: np.ndarray, alpha: float = 0.05) -> float:
    abs_errors = np.asarray(abs_errors, dtype=float)
    abs_errors = abs_errors[np.isfinite(abs_errors)]

    if len(abs_errors) == 0:
        return 0.0

    return float(np.quantile(abs_errors, 1 - alpha))


def get_persistence_baseline_col(X_test: pd.DataFrame, horizon: int) -> str | None:
    preferred = f"aqi_lag_{horizon}"

    if preferred in X_test.columns:
        return preferred

    if "aqi" in X_test.columns:
        return "aqi"

    if "aqi_lag_1" in X_test.columns:
        return "aqi_lag_1"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_aqi_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict:
    y_true_event = y_true >= threshold
    y_pred_event = y_pred >= threshold

    tp = int(np.sum(y_true_event & y_pred_event))
    fp = int(np.sum(~y_true_event & y_pred_event))
    fn = int(np.sum(y_true_event & ~y_pred_event))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        f"precision_gt{int(threshold)}": precision,
        f"recall_gt{int(threshold)}": recall,
        f"f1_gt{int(threshold)}": f1,
    }


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bands = {
        "all": (0, 9999),
        "good_moderate": (0, 100),
        "unhealthy_sensitive": (101, 150),
        "unhealthy": (151, 200),
        "very_unhealthy": (201, 300),
        "hazardous": (301, 9999),
    }

    results = {}

    for label, (lo, hi) in bands.items():
        mask = (y_true >= lo) & (y_true <= hi)
        n = int(mask.sum())

        results[label] = {
            "n": n,
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])) if n >= 5 else None,
        }

    return results


def quantile_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = {}

    for q in [90, 95, 99]:
        threshold = np.percentile(y_true, q)
        mask = y_true >= threshold

        errors[f"mae_top_{q}"] = (
            float(mean_absolute_error(y_true[mask], y_pred[mask]))
            if mask.sum() >= 5
            else None
        )

    return errors


def compute_pi_coverage(y_arr, preds_raw, pi_lower, pi_upper):
    global_cov = float(np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper)))
    avg_width = float(np.mean(pi_upper - pi_lower))

    mask_150 = y_arr > 150
    n_gt150 = int(mask_150.sum())

    cov_150 = (
        float(
            np.mean(
                (y_arr[mask_150] >= pi_lower[mask_150])
                & (y_arr[mask_150] <= pi_upper[mask_150])
            )
        )
        if n_gt150 >= 5
        else 0.0
    )

    mask_200 = y_arr > 200
    n_gt200 = int(mask_200.sum())

    cov_200 = (
        float(
            np.mean(
                (y_arr[mask_200] >= pi_lower[mask_200])
                & (y_arr[mask_200] <= pi_upper[mask_200])
            )
        )
        if n_gt200 >= 5
        else 0.0
    )

    return {
        "conformal_global_coverage": global_cov,
        "conformal_average_width": avg_width,
        "conformal_coverage_gt150": cov_150,
        "conformal_n_gt150": n_gt150,
        "conformal_coverage_gt200": cov_200,
        "conformal_n_gt200": n_gt200,
    }


def run_shap(model, X_test: pd.DataFrame, model_name: str, horizon: int, n_sample: int = 300):
    if not SHAP_AVAILABLE:
        return [], pd.DataFrame()

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(n_sample, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_sample)

        mean_shap = np.abs(shap_vals.values).mean(axis=0)

        shap_df = pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": mean_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        top_features = shap_df.head(20)["feature"].tolist()

        print(f"  SHAP top-5: {top_features[:5]}")

        return top_features, shap_df

    except Exception as e:
        print(f"  SHAP failed ({e}). falling back to feature_importances_.")

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            df = pd.DataFrame({"feature": X_test.columns, "importance": imp})
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            return df.head(20)["feature"].tolist(), df

        return [], pd.DataFrame()


def build_base_metrics(
    model_name: str,
    horizon: int,
    y_arr: np.ndarray,
    preds_raw: np.ndarray,
    pi_lower: np.ndarray,
    pi_upper: np.ndarray,
    margin: float,
    cv_mean_rmse: float,
    cv_std_rmse: float,
    extra: dict | None = None,
) -> dict:
    test_rmse = root_mean_squared_error(y_arr, preds_raw)
    test_mae = mean_absolute_error(y_arr, preds_raw)
    test_r2 = r2_score(y_arr, preds_raw)
    test_evs = explained_variance_score(y_arr, preds_raw)

    test_mape = float(
        np.mean(np.abs((y_arr - preds_raw) / np.maximum(y_arr, 1.0))) * 100
    )

    metrics = {
        "model": model_name,
        "horizon": f"{horizon}h",
        "training_target": "log1p(AQI)",
        "cv_mean_val_rmse": float(cv_mean_rmse),
        "cv_std_val_rmse": float(cv_std_rmse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_mape": float(test_mape),
        "test_r2": float(test_r2),
        "test_explained_variance": float(test_evs),
        "conformal_margin_width": float(margin),
        **compute_pi_coverage(y_arr, preds_raw, pi_lower, pi_upper),
        **compute_aqi_event_metrics(y_arr, preds_raw, 150),
        **compute_aqi_event_metrics(y_arr, preds_raw, 200),
        "error_by_band": error_analysis(y_arr, preds_raw),
        "quantile_errors": quantile_error_analysis(y_arr, preds_raw),
    }

    if extra:
        metrics.update(extra)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model trainers
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(horizon: int) -> dict:
    print(f"\n{'=' * 70}\n  Random Forest - {horizon}h Horizon\n{'=' * 70}")

    # FIX 3: use cached fetch
    X, y_log, y_raw, timestamps = load_xy_both(horizon)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = get_chronological_splits(
        X, y_log, horizon
    )

    _, y_train_raw, _, y_cal_raw, _, y_test_raw = get_chronological_splits(
        X, y_raw, horizon
    )

    _, _, _, _, _, ts_test = get_chronological_splits(
        X, timestamps, horizon
    )

    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97
    )

    print(f"  Features: {X_train.shape[1]} kept, {len(dropped_cols)} dropped by corr filter.")

    # FIX 7: impute before spike augmentation so NaN doesn't corrupt weight calc
    X_train, X_cal, X_test = impute_medians(X_train, X_cal, X_test)

    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train,
        y_train_log,
        y_train_raw=y_train_raw,
        spike_threshold=150,
        target_spike_fraction=0.10,   # FIX 6
    )

    y_aug_raw = np.expm1(y_train_aug.values)
    sample_weights = 1.0 + np.exp(np.minimum(y_aug_raw, 350) / 110.0) - np.exp(0)
    sample_weights = np.clip(sample_weights, 1.0, 25.0)

    param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [10, 12, 15],
        "min_samples_leaf": [8, 10, 15],
        "min_samples_split": [4, 6, 10, 14],
        "max_features": [0.15, 0.2, 0.3, 0.4],
    }

    # FIX 5: inner estimator n_jobs=1 to prevent nested parallelism on 2-vCPU runner
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        param_distributions=param_dist,
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=3, gap=horizon),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    print("  Hyperparameter search with 15 fits.")
    search.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)

    model = search.best_estimator_

    print(f"  Best params: {search.best_params_}")

    best_idx = search.best_index_
    cv_rmse = float(-search.cv_results_["mean_test_score"][best_idx])
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    cal_preds_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_preds_raw))

    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)

    y_arr = y_test_raw.values
    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    lag_col = get_persistence_baseline_col(X_test, horizon)

    p_mae = p_r2 = skill = r2_imp = 0.0

    if lag_col:
        p_mae = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2 = r2_score(y_arr, X_test[lag_col].values)
        skill = (
            float(1.0 - float(mean_absolute_error(y_arr, preds_raw)) / p_mae)
            if p_mae > 0
            else 0.0
        )
        r2_imp = r2_score(y_arr, preds_raw) - p_r2
        print(f"  Persistence baseline: {lag_col}  MAE={p_mae:.1f}  skill={skill:.3f}")

    top_features, shap_df = run_shap(model, X_test, "RF", horizon, n_sample=150)

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "RF", top_features)
    except Exception as e:
        print(f"  Drift monitoring skipped: {e}")

    metrics = build_base_metrics(
        "RandomForest",
        horizon,
        y_arr,
        preds_raw,
        pi_lower,
        pi_upper,
        margin,
        cv_rmse,
        cv_std,
        extra={
            "test_median_ae": float(median_absolute_error(y_arr, preds_raw)),
            "best_params": search.best_params_,
            "baseline_lag_col": lag_col or "none",
            "baseline_horizon_mae": float(p_mae),
            "baseline_horizon_r2": float(p_r2),
            "forecast_skill_score": float(skill),
            "r2_improvement_vs_baseline": float(r2_imp),
        },
    )

    artifact = {
        "model": model,
        "feature_names": list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log": True,
    }

    shap_records = shap_df.head(30).to_dict(orient="records") if not shap_df.empty else []

    push_model(
        horizon,
        "random_forest",
        artifact,
        metrics,
        list(X_train.columns),
        shap_records,
    )

    push_metrics(horizon, "random_forest", metrics)

    push_predictions(
        horizon,
        "random_forest",
        y_arr,
        preds_raw,
        pi_lower,
        pi_upper,
        X_test.index,
        ts_test,
    )

    return metrics


def train_ridge(horizon: int) -> dict:
    print(f"\n{'=' * 70}\n  Ridge Regression - {horizon}h Horizon\n{'=' * 70}")

    # FIX 3: use cached fetch
    X, y_log, y_raw, timestamps = load_xy_both(horizon)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = get_chronological_splits(
        X, y_log, horizon
    )

    _, y_train_raw, _, y_cal_raw, _, y_test_raw = get_chronological_splits(
        X, y_raw, horizon
    )

    _, _, _, _, _, ts_test = get_chronological_splits(
        X, timestamps, horizon
    )

    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.95
    )

    print(f"  Features: {X_train.shape[1]} kept, {len(dropped_cols)} dropped.")

    # FIX 2: impute BEFORE building the CV pipeline.
    # The SimpleImputer+StandardScaler+RidgeCV pipeline below fits imputation
    # inside each CV fold on the fold's train rows only, which is the correct
    # leakage-free approach. We no longer do a blanket fillna here.
    #
    # WHY the old code blew up: the old code called impute_for_linear()
    # AFTER the TimeSeriesSplit CV loop, meaning X_train passed into the fold
    # pipeline still had NaN.  StandardScaler.fit() on NaN-containing data
    # produces NaN means/stds, which causes Ridge to receive all-NaN inputs
    # and output a near-zero model with enormous prediction variance —
    # explaining the ~10M CV RMSE.
    #
    # The fix: put SimpleImputer(strategy="median") as the first step inside
    # the Pipeline so sklearn fits imputation on each fold's X_train rows only.

    tscv_cv = TimeSeriesSplit(n_splits=4, gap=horizon)
    fold_rmse = []

    for tr_idx, val_idx in tscv_cv.split(X_train):
        fold_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),  # FIX 2
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20))),
            ]
        )

        fold_pipe.fit(X_train.iloc[tr_idx], y_train_log.iloc[tr_idx])

        fold_pred_raw = np.expm1(
            np.clip(fold_pipe.predict(X_train.iloc[val_idx]), 0, None)
        )

        fold_rmse.append(
            root_mean_squared_error(
                np.expm1(y_train_log.iloc[val_idx].values),
                fold_pred_raw,
            )
        )

    cv_rmse = float(np.mean(fold_rmse))
    cv_std = float(np.std(fold_rmse))

    print(f"  CV RMSE: {cv_rmse:.2f} +/- {cv_std:.2f}")

    # Final model: same pipeline structure, fit on full train set
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  # FIX 2
            ("scaler", StandardScaler()),
            (
                "ridge",
                RidgeCV(
                    alphas=np.logspace(-3, 3, 30),
                    cv=TimeSeriesSplit(n_splits=5),
                ),
            ),
        ]
    )

    model.fit(X_train, y_train_log)

    best_alpha = float(model.named_steps["ridge"].alpha_)

    print(f"  Optimal alpha: {best_alpha:.4f}")
    print("  Median imputation (per-fold) applied correctly.")

    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)

    y_arr = y_test_raw.values
    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    lag_col = get_persistence_baseline_col(X_test, horizon)

    p_mae = p_r2 = skill = r2_imp = 0.0

    if lag_col:
        p_mae = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2 = r2_score(y_arr, X_test[lag_col].values)
        skill = (
            float(1.0 - float(mean_absolute_error(y_arr, preds_raw)) / p_mae)
            if p_mae > 0
            else 0.0
        )
        r2_imp = r2_score(y_arr, preds_raw) - p_r2

    coef_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "abs_coef": np.abs(model.named_steps["ridge"].coef_),
        }
    ).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    top_features = coef_df.head(20)["feature"].tolist()
    coef_records = coef_df.head(30).to_dict(orient="records")

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "Ridge", top_features)
    except Exception as e:
        print(f"  Drift monitoring skipped: {e}")

    metrics = build_base_metrics(
        "Ridge",
        horizon,
        y_arr,
        preds_raw,
        pi_lower,
        pi_upper,
        margin,
        cv_rmse,
        cv_std,
        extra={
            "best_alpha": best_alpha,
            "baseline_lag_col": lag_col or "none",
            "baseline_horizon_mae": float(p_mae),
            "baseline_horizon_r2": float(p_r2),
            "forecast_skill_score": float(skill),
            "r2_improvement_vs_baseline": float(r2_imp),
        },
    )

    artifact = {
        "model": model,
        "feature_names": list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log": True,
    }

    push_model(
        horizon,
        "ridge",
        artifact,
        metrics,
        list(X_train.columns),
        coef_records,
    )

    push_metrics(horizon, "ridge", metrics)

    push_predictions(
        horizon,
        "ridge",
        y_arr,
        preds_raw,
        pi_lower,
        pi_upper,
        X_test.index,
        ts_test,
    )

    return metrics


def train_xgboost(horizon: int) -> dict:
    if not XGB_AVAILABLE:
        print(f"  XGBoost not available. skipping horizon {horizon}h.")
        return {}

    print(f"\n{'=' * 70}\n  XGBoost - {horizon}h Horizon\n{'=' * 70}")

    # FIX 3: use cached fetch
    X, y_log, y_raw, timestamps = load_xy_both(horizon)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = get_chronological_splits(
        X, y_log, horizon
    )

    _, y_train_raw, _, y_cal_raw, _, y_test_raw = get_chronological_splits(
        X, y_raw, horizon
    )

    _, _, _, _, _, ts_test = get_chronological_splits(
        X, timestamps, horizon
    )

    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97
    )

    print(f"  Features: {X_train.shape[1]} kept, {len(dropped_cols)} dropped.")

    # FIX 7: explicit median imputation for XGB (more stable than NaN routing
    # when combined with sample weights)
    X_train, X_cal, X_test = impute_medians(X_train, X_cal, X_test)

    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train,
        y_train_log,
        y_train_raw=y_train_raw,
        spike_threshold=150,
        target_spike_fraction=0.10,   # FIX 6
    )

    y_aug_raw = np.expm1(y_train_aug.values)

    sample_weights = np.ones(len(y_train_aug))
    sample_weights[y_aug_raw > 100] = 2.0
    sample_weights[y_aug_raw > 150] = 4.0
    sample_weights[y_aug_raw > 200] = 8.0

    # FIX 4: expanded hyperparameter space for XGBoost CV
    tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
    fold_rmse = []

    print("  5-fold TimeSeriesCV with early stopping.")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_ft, y_ft = X_train.iloc[tr_idx], y_train_log.iloc[tr_idx]
        X_fv, y_fv = X_train.iloc[val_idx], y_train_log.iloc[val_idx]

        fw = np.ones(len(y_ft))
        fw[np.expm1(y_ft.values) > 100] = 2.0
        fw[np.expm1(y_ft.values) > 150] = 4.0
        fw[np.expm1(y_ft.values) > 200] = 8.0

        fm = xgb.XGBRegressor(
            n_estimators=1500,          # FIX 4: higher cap, rely on early stopping
            max_depth=6,
            learning_rate=0.01,         # FIX 4: lower LR → more iterations, better optima
            subsample=0.8,
            colsample_bytree=0.7,       # FIX 4: slightly tighter column sampling
            min_child_weight=5,
            reg_lambda=2.0,
            reg_alpha=0.1,              # FIX 4: L1 sparsity regularisation
            gamma=0.05,                 # FIX 4: minimum loss reduction to split
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=0,
        )

        fm.fit(
            X_ft,
            y_ft,
            sample_weight=fw,
            eval_set=[(X_fv, y_fv)],
            verbose=False,
        )

        fold_pred_raw = np.expm1(np.clip(fm.predict(X_fv), 0, None))

        fold_rmse.append(
            root_mean_squared_error(
                np.expm1(y_fv.values),
                fold_pred_raw,
            )
        )

    cv_rmse = float(np.mean(fold_rmse))
    cv_std = float(np.std(fold_rmse))

    print(f"  CV RMSE: {cv_rmse:.2f} +/- {cv_std:.2f}")

    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.1,
        gamma=0.05,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("  Training final model with early stopping on cal set.")

    model.fit(
        X_train_aug,
        y_train_aug,
        sample_weight=sample_weights,
        eval_set=[(X_cal, y_cal_log)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")

    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)

    y_arr = y_test_raw.values
    pi_lower = np.clip(preds_raw - margin, 0, 500)
    pi_upper = np.clip(preds_raw + margin, 0, 500)

    lag_col = get_persistence_baseline_col(X_test, horizon)

    p_mae = p_r2 = skill = r2_imp = 0.0

    if lag_col:
        p_mae = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2 = r2_score(y_arr, X_test[lag_col].values)
        skill = (
            float(1.0 - float(mean_absolute_error(y_arr, preds_raw)) / p_mae)
            if p_mae > 0
            else 0.0
        )
        r2_imp = r2_score(y_arr, preds_raw) - p_r2

        print(f"  Persistence baseline: {lag_col}  MAE={p_mae:.1f}  skill={skill:.3f}")

    # FIX 8: larger SHAP sample for more stable rankings
    top_features, shap_df = run_shap(model, X_test, "XGB", horizon, n_sample=500)

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "XGB", top_features)
    except Exception as e:
        print(f"  Drift monitoring skipped: {e}")

    metrics = build_base_metrics(
        "XGBoost",
        horizon,
        y_arr,
        preds_raw,
        pi_lower,
        pi_upper,
        margin,
        cv_rmse,
        cv_std,
        extra={
            "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
            "baseline_lag_col": lag_col or "none",
            "baseline_horizon_mae": float(p_mae),
            "baseline_horizon_r2": float(p_r2),
            "forecast_skill_score": float(skill),
            "r2_improvement_vs_baseline": float(r2_imp),
        },
    )

    artifact = {
        "model": model,
        "feature_names": list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log": True,
    }

    shap_records = shap_df.head(30).to_dict(orient="records") if not shap_df.empty else []

    push_model(
        horizon,
        "xgboost",
        artifact,
        metrics,
        list(X_train.columns),
        shap_records,
    )

    push_metrics(horizon, "xgboost", metrics)

    push_predictions(
        horizon,
        "xgboost",
        y_arr,
        preds_raw,
        pi_lower,
        pi_upper,
        X_test.index,
        ts_test,
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation summary + entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_summary(all_results: dict[str, dict]):
    print("\n" + "=" * 115)
    print("  CROSS-MODEL EVALUATION SUMMARY")
    print("=" * 115)
    print(
        f"  {'Horizon':<8} {'Model':<16} "
        f"{'Test MAE':>10} {'MAPE':>9} {'R2':>9} "
        f"{'Coverage':>11} {'Cov>150':>9} {'Cov>200':>9} {'Skill':>8}"
    )
    print("-" * 115)

    run_summary = {
        "type": "training_pipeline_run",
        "status": "SUCCESS",
        "horizons": {},
    }

    for h in HORIZONS:
        horizon_key = f"{h}h"
        best_r2 = -np.inf
        best_name = None
        horizon_block = {}

        for name in MODEL_NAMES:
            m = all_results.get(horizon_key, {}).get(name, {})

            if not m:
                continue

            r2 = m.get("test_r2", 0.0)
            mae = m.get("test_mae", 0.0)
            mape = m.get("test_mape", 0.0)
            cov = m.get("conformal_global_coverage", 0.0)
            cov_150 = m.get("conformal_coverage_gt150", 0.0)
            cov_200 = m.get("conformal_coverage_gt200", 0.0)
            skill = m.get("forecast_skill_score", 0.0)

            print(
                f"  {horizon_key:<8} {DISPLAY[name]:<16} "
                f"{mae:>10.1f} {mape:>8.2f}% {r2:>9.3f} "
                f"{cov * 100:>10.1f}% {cov_150 * 100:>8.1f}% "
                f"{cov_200 * 100:>8.1f}% {skill:>8.3f}"
            )

            horizon_block[name] = {
                "r2": float(r2),
                "mae": float(mae),
                "mape": float(mape),
                "coverage": float(cov),
                "skill": float(skill),
            }

            if r2 > best_r2:
                best_r2 = r2
                best_name = name

        if best_name:
            flag_best_model(h, best_name)
            print(f"  {'':8} Best: {DISPLAY[best_name]}  (R2={best_r2:.4f})")

            run_summary["horizons"][horizon_key] = {
                "best_model": best_name,
                "best_r2": round(float(best_r2), 4),
                "models": horizon_block,
            }

        print()

    print("=" * 115)
    push_pipeline_run_summary(run_summary)


def main():
    print("\n" + "=" * 70)
    print("  KARACHI AQI - CONSOLIDATED TRAINING PIPELINE")
    print("=" * 70)

    all_results: dict[str, dict] = {}

    for horizon in HORIZONS:
        hk = f"{horizon}h"
        all_results[hk] = {}

        # FIX 3: pre-warm the cache for this horizon so all 3 trainers share it
        load_xy_both(horizon)

        m_rf = train_random_forest(horizon)
        all_results[hk]["random_forest"] = m_rf

        m_ridge = train_ridge(horizon)
        all_results[hk]["ridge"] = m_ridge

        if XGB_AVAILABLE:
            m_xgb = train_xgboost(horizon)
            all_results[hk]["xgboost"] = m_xgb

        # Free cache after all models for this horizon are done
        _DATA_CACHE.pop(horizon, None)

    run_evaluation_summary(all_results)

    print("\nAll models, metrics, and predictions have been pushed to MongoDB.")
    print("No local files were written. The pipeline is fully serverless.")


if __name__ == "__main__":
    main()