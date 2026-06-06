"""
train_pipeline.py

Improved training pipeline for AQI Predictor Karachi.

Key fixes:
1. Uses current AQI, PM2.5, PM10, weather, and pollutants as valid causal features.
2. Drops only future target columns and unsafe metadata.
3. Trains stronger models with less underfitting.
4. Adds Extra Trees and a weighted ensemble.
5. Uses raw AQI target instead of log target.
6. Saves all artifacts, metrics, predictions, and best model flags to MongoDB.
"""

import io
import os
import sys
import json
import math
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import gridfs
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
)

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DB_NAME = "karachi_aqi"
FEATURE_COLLECTION = "processed_features"

HORIZONS = [12, 24, 48, 72]

RANDOM_STATE = 42
TEST_SIZE = 0.15
CAL_SIZE = 0.15

HIGH_MISSING_THRESHOLD = 0.35
CORRELATION_THRESHOLD = 0.995

JOBLIB_COMPRESS = 3


MODEL_LABELS = {
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "xgboost": "XGBoost",
    "weighted_ensemble": "Weighted Ensemble",
}


def log(message: str) -> None:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", flush=True)


def get_db():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is not set.")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client[DB_NAME], client


def clean_for_mongo(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.ndarray, list, tuple)):
        return [clean_for_mongo(v) for v in value]

    if isinstance(value, dict):
        return {k: clean_for_mongo(v) for k, v in value.items()}

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def load_feature_store() -> pd.DataFrame:
    db, client = get_db()

    try:
        docs = list(db[FEATURE_COLLECTION].find({}, {"_id": 0}))
    except PyMongoError as error:
        client.close()
        raise RuntimeError(f"Failed to read MongoDB feature store: {error}")

    client.close()

    if not docs:
        raise RuntimeError(
            f"MongoDB collection '{DB_NAME}.{FEATURE_COLLECTION}' is empty."
        )

    df = pd.DataFrame(docs)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    else:
        raise KeyError("Feature store is missing datetime or timestamp column.")

    df = df.sort_values("datetime").reset_index(drop=True)

    log(f"Feature store loaded: {len(df):,} rows and {len(df.columns):,} columns")
    log(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def get_target_column(horizon: int) -> str:
    target_col = f"target_aqi_{horizon}h"
    return target_col


def is_forbidden_feature(column: str) -> bool:
    lower = column.lower()

    if lower in ["datetime", "timestamp"]:
        return True

    if lower == "_id":
        return True

    if lower.startswith("target_"):
        return True

    if lower in [
        "prediction",
        "predicted",
        "actual",
        "error",
        "residual",
        "row_index",
    ]:
        return True

    return False


def add_runtime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])

        df["runtime_hour"] = dt.dt.hour
        df["runtime_month"] = dt.dt.month
        df["runtime_dayofyear"] = dt.dt.dayofyear
        df["runtime_weekday"] = dt.dt.weekday

        df["runtime_hour_sin"] = np.sin(2 * np.pi * df["runtime_hour"] / 24)
        df["runtime_hour_cos"] = np.cos(2 * np.pi * df["runtime_hour"] / 24)
        df["runtime_month_sin"] = np.sin(2 * np.pi * df["runtime_month"] / 12)
        df["runtime_month_cos"] = np.cos(2 * np.pi * df["runtime_month"] / 12)
        df["runtime_doy_sin"] = np.sin(2 * np.pi * df["runtime_dayofyear"] / 365.25)
        df["runtime_doy_cos"] = np.cos(2 * np.pi * df["runtime_dayofyear"] / 365.25)

    if "aqi" in df.columns:
        for lag in [1, 3, 6, 12, 24, 48, 72]:
            lag_col = f"aqi_lag_{lag}"
            if lag_col in df.columns:
                df[f"aqi_vs_lag_{lag}"] = df["aqi"] - df[lag_col]

    if "pm25" in df.columns:
        for lag in [1, 3, 6, 12, 24, 48, 72]:
            lag_col = f"pm25_lag_{lag}"
            if lag_col in df.columns:
                df[f"pm25_vs_lag_{lag}"] = df["pm25"] - df[lag_col]

    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm25_to_pm10_live_ratio"] = df["pm25"] / (df["pm10"].replace(0, np.nan))

    if "aqi" in df.columns and "pm25" in df.columns:
        df["aqi_pm25_interaction"] = df["aqi"] * df["pm25"]

    if "pm25" in df.columns and "humidity" in df.columns:
        df["pm25_humidity_live_interaction"] = df["pm25"] * df["humidity"]

    if "pm25" in df.columns and "wind_speed" in df.columns:
        df["pm25_wind_live_interaction"] = df["pm25"] / (df["wind_speed"].abs() + 1)

    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_live_interaction"] = df["temperature"] * df["humidity"]

    return df


def build_xy(df: pd.DataFrame, horizon: int):
    target_col = get_target_column(horizon)

    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    work = df.copy()
    work = add_runtime_features(work)

    work = work.dropna(subset=[target_col]).reset_index(drop=True)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).reset_index(drop=True)

    work[target_col] = work[target_col].clip(lower=0, upper=500)

    y = work[target_col].astype(float)

    drop_cols = [col for col in work.columns if is_forbidden_feature(col)]
    X = work.drop(columns=drop_cols, errors="ignore")

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_fraction = X.isna().mean()
    keep_cols = missing_fraction[missing_fraction <= HIGH_MISSING_THRESHOLD].index.tolist()
    X = X[keep_cols]

    nunique = X.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = X[keep_cols]

    X = remove_highly_correlated_features(X)

    log(f"Horizon {horizon}h target: {target_col}")
    log(f"Rows after target cleaning: {len(X):,}")
    log(f"Feature count after cleaning: {X.shape[1]:,}")

    if "aqi" in X.columns:
        corr = pd.Series(X["aqi"]).corr(y)
        log(f"Current AQI correlation with target {horizon}h: {corr:.3f}")

    baseline_col = get_baseline_col(X, horizon)
    if baseline_col:
        corr = pd.Series(X[baseline_col]).corr(y)
        log(f"{baseline_col} correlation with target {horizon}h: {corr:.3f}")

    return X, y, work["datetime"]


def remove_highly_correlated_features(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] <= 2:
        return X

    protected_terms = [
        "aqi",
        "pm25",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "temperature",
        "humidity",
        "pressure",
        "wind",
        "lag",
        "roll",
        "ewm",
        "fourier",
        "runtime",
        "month",
        "doy",
    ]

    protected = set()
    for col in X.columns:
        lower = col.lower()
        if any(term in lower for term in protected_terms):
            protected.add(col)

    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = []
    for col in upper.columns:
        if col in protected:
            continue

        high_corr = upper[col][upper[col] > CORRELATION_THRESHOLD]
        if len(high_corr) > 0:
            drop_cols.append(col)

    if drop_cols:
        log(f"Dropping {len(drop_cols):,} highly correlated non-protected features")

    return X.drop(columns=drop_cols, errors="ignore")


def chronological_split(X, y, dt):
    n = len(X)

    test_start = int(n * (1 - TEST_SIZE))
    cal_start = int(test_start * (1 - CAL_SIZE))

    X_train = X.iloc[:cal_start].copy()
    y_train = y.iloc[:cal_start].copy()

    X_cal = X.iloc[cal_start:test_start].copy()
    y_cal = y.iloc[cal_start:test_start].copy()

    X_test = X.iloc[test_start:].copy()
    y_test = y.iloc[test_start:].copy()

    dt_test = dt.iloc[test_start:].copy()

    log(f"Train rows: {len(X_train):,}")
    log(f"Calibration rows: {len(X_cal):,}")
    log(f"Test rows: {len(X_test):,}")

    return X_train, y_train, X_cal, y_cal, X_test, y_test, dt_test


def get_baseline_col(X: pd.DataFrame, horizon: int) -> str | None:
    exact = f"aqi_lag_{horizon}"
    if exact in X.columns:
        return exact

    if "aqi" in X.columns:
        return "aqi"

    fallback_cols = [
        "aqi_lag_24",
        "aqi_lag_12",
        "aqi_lag_48",
        "aqi_lag_72",
        "pm25",
        "pm25_lag_24",
    ]

    for col in fallback_cols:
        if col in X.columns:
            return col

    return None


def make_model_builders():
    alphas = np.logspace(-3, 5, 40)

    builders = {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", RidgeCV(alphas=alphas)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.05, 0.1, 0.2, 0.4, 0.7],
                        alphas=np.logspace(-4, 2, 40),
                        max_iter=12000,
                        cv=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        max_depth=28,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        max_features=0.65,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=450,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features=0.85,
                        bootstrap=False,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.035,
                        max_iter=850,
                        max_leaf_nodes=31,
                        min_samples_leaf=18,
                        l2_regularization=0.03,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if XGB_AVAILABLE:
        builders["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    xgb.XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=1400,
                        learning_rate=0.02,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.82,
                        colsample_bytree=0.82,
                        reg_alpha=0.15,
                        reg_lambda=4.0,
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return builders


def calculate_metrics(y_true, y_pred, baseline_pred=None, pi_lower=None, pi_upper=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    denom = np.where(np.abs(y_true) < 1, 1, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_medae": float(medae),
        "test_r2": float(r2),
        "test_mape": float(mape),
        "explained_variance": float(evs),
    }

    if baseline_pred is not None:
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        skill = 1 - (mae / baseline_mae) if baseline_mae > 0 else 0
        metrics["persistence_baseline_mae"] = float(baseline_mae)
        metrics["forecast_skill_score"] = float(skill)

    if pi_lower is not None and pi_upper is not None:
        coverage = np.mean((y_true >= pi_lower) & (y_true <= pi_upper))
        width = np.mean(pi_upper - pi_lower)

        mask_150 = y_true > 150
        mask_200 = y_true > 200

        metrics["conformal_global_coverage"] = float(coverage)
        metrics["conformal_average_width"] = float(width)
        metrics["conformal_coverage_gt150"] = (
            float(np.mean((y_true[mask_150] >= pi_lower[mask_150]) & (y_true[mask_150] <= pi_upper[mask_150])))
            if mask_150.sum() >= 5
            else 0.0
        )
        metrics["conformal_coverage_gt200"] = (
            float(np.mean((y_true[mask_200] >= pi_lower[mask_200]) & (y_true[mask_200] <= pi_upper[mask_200])))
            if mask_200.sum() >= 5
            else 0.0
        )
        metrics["conformal_n_gt150"] = int(mask_150.sum())
        metrics["conformal_n_gt200"] = int(mask_200.sum())

    return metrics


def prediction_interval_from_calibration(y_cal, pred_cal, pred_test, alpha=0.05):
    residuals = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(pred_cal, dtype=float))
    margin = float(np.quantile(residuals, 1 - alpha))

    pred_test = np.asarray(pred_test, dtype=float)
    lower = np.clip(pred_test - margin, 0, 500)
    upper = np.clip(pred_test + margin, 0, 500)

    return margin, lower, upper


def get_top_features(model, feature_names, top_n=20):
    try:
        final_model = model.named_steps.get("model")

        if hasattr(final_model, "feature_importances_"):
            values = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            values = np.abs(final_model.coef_)
        else:
            return []

        pairs = list(zip(feature_names, values))
        pairs = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)

        return [
            {"feature": str(name), "importance": float(value)}
            for name, value in pairs[:top_n]
        ]

    except Exception:
        return []


def serialise_model_artifact(model, feature_names, conformal_margin, metrics, top_features):
    artifact = {
        "model": model,
        "feature_names": list(feature_names),
        "use_log": False,
        "conformal_margin": float(conformal_margin),
        "metrics": clean_for_mongo(metrics),
        "top_features": clean_for_mongo(top_features),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=JOBLIB_COMPRESS)
    return buffer.getvalue()


def save_model_artifact(horizon, model_name, model, feature_names, metrics, conformal_margin, top_features):
    db, client = get_db()

    registry = db["model_registry"]
    fs = gridfs.GridFS(db, collection="model_artifacts")

    existing = registry.find_one(
        {"horizon": horizon, "model_name": model_name},
        {"artifact_gridfs_id": 1},
    )

    if existing and existing.get("artifact_gridfs_id"):
        try:
            fs.delete(existing["artifact_gridfs_id"])
            log(f"Deleted old GridFS artifact for {model_name} {horizon}h")
        except Exception as error:
            log(f"Old artifact cleanup skipped for {model_name} {horizon}h: {error}")

    model_bytes = serialise_model_artifact(
        model=model,
        feature_names=feature_names,
        conformal_margin=conformal_margin,
        metrics=metrics,
        top_features=top_features,
    )

    trained_at = datetime.now(tz=timezone.utc).isoformat()
    filename = f"{model_name}_{horizon}h_{trained_at}.joblib"

    artifact_id = fs.put(
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

    registry_doc = {
        "horizon": int(horizon),
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "trained_at": trained_at,
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "artifact_storage": "gridfs",
        "artifact_gridfs_id": artifact_id,
        "artifact_filename": filename,
        "artifact_size_bytes": len(model_bytes),
        "top_features": clean_for_mongo(top_features),
        **clean_for_mongo(metrics),
    }

    registry.update_one(
        {"horizon": horizon, "model_name": model_name},
        {"$set": registry_doc, "$unset": {"model_binary": ""}},
        upsert=True,
    )

    client.close()

    log(
        f"Saved model artifact: {model_name} {horizon}h "
        f"size={len(model_bytes) / 1024 / 1024:.2f}MB"
    )


def save_metrics(horizon, model_name, metrics):
    db, client = get_db()

    doc = {
        "horizon": int(horizon),
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "logged_at": datetime.now(tz=timezone.utc).isoformat(),
        **clean_for_mongo(metrics),
    }

    db["model_metrics"].insert_one(doc)
    client.close()


def save_predictions(horizon, model_name, y_true, y_pred, pi_lower, pi_upper, dt_test):
    db, client = get_db()
    col = db["model_predictions"]

    col.delete_many({"horizon": horizon, "model_name": model_name})

    docs = []
    for index, actual in enumerate(y_true):
        docs.append(
            {
                "horizon": int(horizon),
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "row_index": int(index),
                "datetime": clean_for_mongo(dt_test.iloc[index]) if index < len(dt_test) else None,
                "actual": float(actual),
                "predicted": float(y_pred[index]),
                "pi_lower": float(pi_lower[index]),
                "pi_upper": float(pi_upper[index]),
            }
        )

    for start in range(0, len(docs), 1000):
        col.insert_many(docs[start:start + 1000], ordered=False)

    client.close()

    log(f"Saved predictions: {model_name} {horizon}h rows={len(docs):,}")


def flag_best_model(horizon, best_model_name):
    db, client = get_db()

    col = db["model_registry"]
    col.update_many({"horizon": horizon}, {"$set": {"is_best": False}})
    col.update_one(
        {"horizon": horizon, "model_name": best_model_name},
        {"$set": {"is_best": True}},
    )

    client.close()


def save_pipeline_summary(summary):
    db, client = get_db()

    db["pipeline_runs"].insert_one(
        {
            "run_at": datetime.now(tz=timezone.utc).isoformat(),
            **clean_for_mongo(summary),
        }
    )

    client.close()


def train_single_horizon(df: pd.DataFrame, horizon: int):
    log("=" * 80)
    log(f"Training horizon: {horizon}h")
    log("=" * 80)

    X, y, dt = build_xy(df, horizon)

    log("Target distribution:")
    print(y.describe().round(2).to_string(), flush=True)

    X_train, y_train, X_cal, y_cal, X_test, y_test, dt_test = chronological_split(X, y, dt)

    baseline_col = get_baseline_col(X, horizon)

    if baseline_col:
        baseline_test = X_test[baseline_col].fillna(X_train[baseline_col].median()).clip(0, 500).values
    else:
        baseline_test = np.full(len(y_test), y_train.median())

    builders = make_model_builders()

    fitted_initial = {}
    cal_scores = {}
    results = {}

    for model_name, model_template in builders.items():
        log("-" * 80)
        log(f"Initial calibration training: {MODEL_LABELS.get(model_name, model_name)}")

        model = clone(model_template)
        model.fit(X_train, y_train)

        pred_cal = np.clip(model.predict(X_cal), 0, 500)
        cal_mae = mean_absolute_error(y_cal, pred_cal)
        cal_r2 = r2_score(y_cal, pred_cal)

        fitted_initial[model_name] = model
        cal_scores[model_name] = {
            "cal_mae": float(cal_mae),
            "cal_r2": float(cal_r2),
            "pred_cal": pred_cal,
        }

        log(f"Calibration MAE: {cal_mae:.3f}")
        log(f"Calibration R2: {cal_r2:.4f}")

    X_train_full = pd.concat([X_train, X_cal], axis=0)
    y_train_full = pd.concat([y_train, y_cal], axis=0)

    final_models = {}

    for model_name, model_template in builders.items():
        log("-" * 80)
        log(f"Final training: {MODEL_LABELS.get(model_name, model_name)}")

        final_model = clone(model_template)
        final_model.fit(X_train_full, y_train_full)
        final_models[model_name] = final_model

        pred_cal = cal_scores[model_name]["pred_cal"]
        pred_test = np.clip(final_model.predict(X_test), 0, 500)

        margin, pi_lower, pi_upper = prediction_interval_from_calibration(
            y_cal=y_cal,
            pred_cal=pred_cal,
            pred_test=pred_test,
            alpha=0.05,
        )

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=pred_test,
            baseline_pred=baseline_test,
            pi_lower=pi_lower,
            pi_upper=pi_upper,
        )

        metrics["cal_mae"] = cal_scores[model_name]["cal_mae"]
        metrics["cal_r2"] = cal_scores[model_name]["cal_r2"]
        metrics["conformal_margin"] = margin
        metrics["feature_count"] = X.shape[1]
        metrics["target_column"] = get_target_column(horizon)
        metrics["baseline_column"] = baseline_col

        top_features = get_top_features(final_model, X.columns.tolist())

        save_model_artifact(
            horizon=horizon,
            model_name=model_name,
            model=final_model,
            feature_names=X.columns.tolist(),
            metrics=metrics,
            conformal_margin=margin,
            top_features=top_features,
        )

        save_metrics(horizon, model_name, metrics)
        save_predictions(horizon, model_name, y_test.values, pred_test, pi_lower, pi_upper, dt_test)

        results[model_name] = {
            "metrics": metrics,
            "pred_test": pred_test,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
            "model": final_model,
        }

        log(
            f"{MODEL_LABELS.get(model_name, model_name)} "
            f"MAE={metrics['test_mae']:.3f} "
            f"RMSE={metrics['test_rmse']:.3f} "
            f"R2={metrics['test_r2']:.4f} "
            f"Skill={metrics.get('forecast_skill_score', 0):.4f}"
        )

    ensemble_result = train_weighted_ensemble(
        builders=builders,
        cal_scores=cal_scores,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        baseline_test=baseline_test,
        horizon=horizon,
        feature_names=X.columns.tolist(),
        dt_test=dt_test,
        baseline_col=baseline_col,
    )

    if ensemble_result:
        results["weighted_ensemble"] = ensemble_result

    best_model_name = max(
        results.keys(),
        key=lambda name: results[name]["metrics"]["test_r2"],
    )

    flag_best_model(horizon, best_model_name)

    log("=" * 80)
    log(
        f"Best model for {horizon}h: "
        f"{MODEL_LABELS.get(best_model_name, best_model_name)} "
        f"R2={results[best_model_name]['metrics']['test_r2']:.4f}"
    )
    log("=" * 80)

    return {
        "horizon": horizon,
        "best_model": best_model_name,
        "best_r2": results[best_model_name]["metrics"]["test_r2"],
        "models": {
            name: clean_for_mongo(result["metrics"])
            for name, result in results.items()
        },
    }


def train_weighted_ensemble(
    builders,
    cal_scores,
    X_train_full,
    y_train_full,
    X_cal,
    y_cal,
    X_test,
    y_test,
    baseline_test,
    horizon,
    feature_names,
    dt_test,
    baseline_col,
):
    usable_models = []

    for model_name, score in cal_scores.items():
        cal_mae = score["cal_mae"]

        if not np.isfinite(cal_mae):
            continue

        usable_models.append((model_name, cal_mae))

    if len(usable_models) < 2:
        log("Skipping weighted ensemble because fewer than 2 models are usable")
        return None

    usable_models = sorted(usable_models, key=lambda item: item[1])
    usable_models = usable_models[:5]

    weights = []
    estimators = []

    for model_name, cal_mae in usable_models:
        weight = 1 / max(cal_mae, 1e-6) ** 2
        weights.append(weight)
        estimators.append((model_name, clone(builders[model_name])))

    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    log("-" * 80)
    log("Final training: Weighted Ensemble")
    log(f"Ensemble models: {[name for name, _ in usable_models]}")
    log(f"Ensemble weights: {[round(w, 3) for w in weights]}")

    ensemble = VotingRegressor(
        estimators=estimators,
        weights=weights,
        n_jobs=-1,
    )

    ensemble.fit(X_train_full, y_train_full)

    cal_pred_matrix = []
    for model_name, _ in usable_models:
        pred_cal = cal_scores[model_name]["pred_cal"]
        cal_pred_matrix.append(pred_cal)

    pred_cal_ensemble = np.average(
        np.vstack(cal_pred_matrix),
        axis=0,
        weights=weights,
    )

    pred_test = np.clip(ensemble.predict(X_test), 0, 500)

    margin, pi_lower, pi_upper = prediction_interval_from_calibration(
        y_cal=y_cal,
        pred_cal=pred_cal_ensemble,
        pred_test=pred_test,
        alpha=0.05,
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=pred_test,
        baseline_pred=baseline_test,
        pi_lower=pi_lower,
        pi_upper=pi_upper,
    )

    metrics["cal_mae"] = float(mean_absolute_error(y_cal, pred_cal_ensemble))
    metrics["cal_r2"] = float(r2_score(y_cal, pred_cal_ensemble))
    metrics["conformal_margin"] = margin
    metrics["feature_count"] = len(feature_names)
    metrics["target_column"] = get_target_column(horizon)
    metrics["baseline_column"] = baseline_col
    metrics["ensemble_members"] = [name for name, _ in usable_models]
    metrics["ensemble_weights"] = weights

    top_features = []

    save_model_artifact(
        horizon=horizon,
        model_name="weighted_ensemble",
        model=ensemble,
        feature_names=feature_names,
        metrics=metrics,
        conformal_margin=margin,
        top_features=top_features,
    )

    save_metrics(horizon, "weighted_ensemble", metrics)
    save_predictions(horizon, "weighted_ensemble", y_test.values, pred_test, pi_lower, pi_upper, dt_test)

    log(
        f"Weighted Ensemble "
        f"MAE={metrics['test_mae']:.3f} "
        f"RMSE={metrics['test_rmse']:.3f} "
        f"R2={metrics['test_r2']:.4f} "
        f"Skill={metrics.get('forecast_skill_score', 0):.4f}"
    )

    return {
        "metrics": metrics,
        "pred_test": pred_test,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
        "model": ensemble,
    }


def print_summary(all_results):
    log("=" * 110)
    log("Cross model evaluation summary")
    log("=" * 110)

    header = (
        f"{'Horizon':<8}"
        f"{'Model':<28}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'R2':>10}"
        f"{'MAPE':>10}"
        f"{'Coverage':>12}"
        f"{'Skill':>10}"
    )
    print(header, flush=True)
    print("-" * 110, flush=True)

    for horizon_result in all_results:
        horizon = horizon_result["horizon"]

        for model_name, metrics in horizon_result["models"].items():
            coverage = metrics.get("conformal_global_coverage")
            coverage_pct = coverage * 100 if coverage is not None else 0

            row = (
                f"{str(horizon) + 'h':<8}"
                f"{MODEL_LABELS.get(model_name, model_name):<28}"
                f"{metrics.get('test_mae', 0):>10.2f}"
                f"{metrics.get('test_rmse', 0):>10.2f}"
                f"{metrics.get('test_r2', 0):>10.4f}"
                f"{metrics.get('test_mape', 0):>9.2f}%"
                f"{coverage_pct:>11.1f}%"
                f"{metrics.get('forecast_skill_score', 0):>10.4f}"
            )
            print(row, flush=True)

        print(
            f"Best for {horizon}h: "
            f"{MODEL_LABELS.get(horizon_result['best_model'], horizon_result['best_model'])} "
            f"R2={horizon_result['best_r2']:.4f}",
            flush=True,
        )
        print("", flush=True)


def main():
    log("=" * 80)
    log("Karachi AQI improved training pipeline")
    log("=" * 80)

    df = load_feature_store()

    all_results = []

    for horizon in HORIZONS:
        target_col = get_target_column(horizon)

        if target_col not in df.columns:
            log(f"Skipping {horizon}h because {target_col} is missing")
            continue

        result = train_single_horizon(df, horizon)
        all_results.append(result)

    print_summary(all_results)

    summary = {
        "pipeline": "training",
        "version": "improved_current_feature_ensemble_v1",
        "horizons": all_results,
        "notes": [
            "Current AQI, pollutants, and weather are retained as causal prediction features.",
            "Only future target columns are removed.",
            "Models are trained on chronological splits.",
            "Best model is selected by test R2.",
        ],
    }

    save_pipeline_summary(summary)

    log("Training complete. Models, metrics, predictions, and run summary saved to MongoDB.")


if __name__ == "__main__":
    main()