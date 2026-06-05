"""
train_pipeline.py
-----------------
Consolidated training pipeline for Karachi AQI forecasting.

Replaces:  train_random_forest.py  +  train_ridge.py  +  train_xgboost.py  +  evaluate.py

All model artefacts, metrics, predictions, feature importances, and the
final evaluation summary are written ONLY to MongoDB — no local files.
This keeps the pipeline fully serverless and automation-friendly (GitHub Actions,
cron jobs, etc. can run this with nothing but MONGODB_URI in the environment).

MongoDB collections written:
  karachi_aqi.model_registry      — serialised model binary + metadata per horizon
  karachi_aqi.model_metrics        — full metrics dict per model × horizon
  karachi_aqi.model_predictions    — actual vs predicted + PI bounds per row
  karachi_aqi.pipeline_runs        — one summary document per full pipeline run

HOW MODELS ARE STORED / LOADED:
  joblib.dump() → BytesIO buffer → Binary(buffer.getvalue()) stored in MongoDB.
  To load:  artifact = col.find_one({"horizon": h, "model_name": name, "is_best": True})
            model_obj = joblib.load(BytesIO(artifact["model_binary"]))["model"]

STRUCTURE:
  Section 1 — MongoDB helpers
  Section 2 — Shared analysis helpers (error bands, quantile errors, SHAP, conformal)
  Section 3 — train_random_forest(horizon)
  Section 4 — train_ridge(horizon)
  Section 5 — train_xgboost(horizon)
  Section 6 — run_evaluation_summary(all_results)
  Section 7 — main()
"""

import io
import os
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from bson import Binary

from sklearn.ensemble import RandomForestRegressor
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
    print("WARNING: xgboost not installed — XGBoost will be skipped. pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed — SHAP explanations will be skipped. pip install shap")

from load_data import (
    load_xy,
    get_chronological_splits,
    get_spike_augmented_train,
    apply_leakage_free_correlation_filter,
    calculate_conformal_margin,
    compute_aqi_event_metrics,
    get_persistence_baseline_col,
    impute_for_linear,
)

try:
    from monitoring import run_data_drift_monitoring
except ImportError:
    def run_data_drift_monitoring(*args, **kwargs):
        print("  [monitoring] skipped — module not found.")

HORIZONS = [24, 48, 72]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MongoDB Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_db():
    """Returns an authenticated MongoDB database handle."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set.")
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client["karachi_aqi"], client


def _serialise_model(artifact: dict) -> Binary:
    """Serialise a joblib artifact dict to a BSON-safe Binary blob."""
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    return Binary(buf.getvalue())


def push_model(
    horizon: int,
    model_name: str,
    artifact: dict,
    metrics: dict,
    feature_names: list,
    top_features: list | None = None,
):
    """
    Upsert a trained model + its full metrics into model_registry.
    The model binary is stored as a BSON Binary field so no filesystem is needed.
    """
    db, client = _get_db()
    col = db["model_registry"]

    doc = {
        "horizon":       horizon,
        "model_name":    model_name,
        "trained_at":    datetime.now(tz=timezone.utc).isoformat(),
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "model_binary":  _serialise_model(artifact),
        **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
    }
    if top_features:
        doc["top_features"] = top_features

    col.update_one(
        {"horizon": horizon, "model_name": model_name},
        {"$set": doc},
        upsert=True,
    )
    client.close()
    print(f"  [MongoDB] model_registry ← {model_name} horizon={horizon}h")


def push_metrics(horizon: int, model_name: str, metrics: dict):
    """Insert a full metrics document into model_metrics (append-only history)."""
    db, client = _get_db()
    db["model_metrics"].insert_one({
        "horizon":    horizon,
        "model_name": model_name,
        "logged_at":  datetime.now(tz=timezone.utc).isoformat(),
        **metrics,
    })
    client.close()
    print(f"  [MongoDB] model_metrics ← {model_name} horizon={horizon}h")


def push_predictions(
    horizon: int,
    model_name: str,
    y_arr: np.ndarray,
    preds_raw: np.ndarray,
    pi_lower: np.ndarray,
    pi_upper: np.ndarray,
    index: pd.Index,
):
    """Bulk-upsert test-set predictions into model_predictions."""
    db, client = _get_db()
    col = db["model_predictions"]
    ops = [
        UpdateOne(
            {"horizon": horizon, "model_name": model_name, "row_index": int(idx)},
            {"$set": {
                "horizon":    horizon,
                "model_name": model_name,
                "row_index":  int(idx),
                "actual":     float(y_arr[i]),
                "predicted":  float(preds_raw[i]),
                "pi_lower":   float(pi_lower[i]),
                "pi_upper":   float(pi_upper[i]),
            }},
            upsert=True,
        )
        for i, idx in enumerate(index)
    ]
    for i in range(0, len(ops), 1000):
        col.bulk_write(ops[i:i+1000], ordered=False)
    client.close()
    print(f"  [MongoDB] model_predictions ← {model_name} horizon={horizon}h "
          f"({len(ops)} rows)")


def flag_best_model(horizon: int, best_model_name: str):
    """Set is_best=True on the winner and False on all other models for this horizon."""
    db, client = _get_db()
    col = db["model_registry"]
    col.update_many({"horizon": horizon}, {"$set": {"is_best": False}})
    col.update_one(
        {"horizon": horizon, "model_name": best_model_name},
        {"$set": {"is_best": True}},
    )
    client.close()


def push_pipeline_run_summary(summary: dict):
    """Append one summary document per full pipeline execution."""
    db, client = _get_db()
    db["pipeline_runs"].insert_one({
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        **summary,
    })
    client.close()
    print("  [MongoDB] pipeline_runs ← run summary saved.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Shared Analysis Helpers
# ══════════════════════════════════════════════════════════════════════════════

def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-AQI-band MAE breakdown (Good/Moderate/USG/Unhealthy/Very/Hazardous)."""
    bands = {
        "all":                 (0,    9999),
        "good_moderate":       (0,    100),
        "unhealthy_sensitive": (101,  150),
        "unhealthy":           (151,  200),
        "very_unhealthy":      (201,  300),
        "hazardous":           (301,  9999),
    }
    results = {}
    for label, (lo, hi) in bands.items():
        mask = (y_true >= lo) & (y_true <= hi)
        n    = int(mask.sum())
        results[label] = {
            "n":   n,
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
            if mask.sum() >= 5 else None
        )
    return errors


def compute_pi_coverage(y_arr, preds_raw, pi_lower, pi_upper):
    """Returns global + spike-tier conformal prediction interval coverage stats."""
    global_cov  = float(np.mean((y_arr >= pi_lower) & (y_arr <= pi_upper)))
    avg_width   = float(np.mean(pi_upper - pi_lower))

    mask_150 = y_arr > 150; n_gt150 = int(mask_150.sum())
    cov_150  = float(np.mean(
        (y_arr[mask_150] >= pi_lower[mask_150]) & (y_arr[mask_150] <= pi_upper[mask_150])
    )) if n_gt150 >= 5 else 0.0

    mask_200 = y_arr > 200; n_gt200 = int(mask_200.sum())
    cov_200  = float(np.mean(
        (y_arr[mask_200] >= pi_lower[mask_200]) & (y_arr[mask_200] <= pi_upper[mask_200])
    )) if n_gt200 >= 5 else 0.0

    return {
        "conformal_global_coverage":  global_cov,
        "conformal_average_width":    avg_width,
        "conformal_coverage_gt150":   cov_150,
        "conformal_n_gt150":          n_gt150,
        "conformal_coverage_gt200":   cov_200,
        "conformal_n_gt200":          n_gt200,
    }


def run_shap(model, X_test: pd.DataFrame, model_name: str, horizon: int, n_sample: int = 200):
    """
    Computes SHAP values on a random subsample of the test set.
    Returns (top_feature_list, mean_abs_shap_df).
    """
    if not SHAP_AVAILABLE:
        return [], pd.DataFrame()

    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(n_sample, len(X_test)), replace=False)
    X_sample   = X_test.iloc[sample_idx]

    try:
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer(X_sample)
        mean_shap   = np.abs(shap_vals.values).mean(axis=0)
        shap_df     = pd.DataFrame({
            "feature":       X_sample.columns,
            "mean_abs_shap": mean_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        top_features = shap_df.head(20)["feature"].tolist()
        print(f"  SHAP top-5: {top_features[:5]}")
        return top_features, shap_df
    except Exception as e:
        print(f"  SHAP failed ({e}) — falling back to feature_importances_.")
        imp  = model.feature_importances_
        df   = pd.DataFrame({"feature": X_test.columns, "importance": imp})
        df   = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(20)["feature"].tolist(), df


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
    """Assembles the standard metrics dict shared by all three models."""
    test_rmse = root_mean_squared_error(y_arr, preds_raw)
    test_mae  = mean_absolute_error(y_arr, preds_raw)
    test_r2   = r2_score(y_arr, preds_raw)
    test_evs  = explained_variance_score(y_arr, preds_raw)
    test_mape = float(np.mean(np.abs((y_arr - preds_raw) / np.maximum(y_arr, 1.0))) * 100)

    metrics = {
        "model":                   model_name,
        "horizon":                 f"{horizon}h",
        "training_target":         "log1p(AQI)",
        "cv_mean_val_rmse":        float(cv_mean_rmse),
        "cv_std_val_rmse":         float(cv_std_rmse),
        "test_rmse":               float(test_rmse),
        "test_mae":                float(test_mae),
        "test_mape":               float(test_mape),
        "test_r2":                 float(test_r2),
        "test_explained_variance": float(test_evs),
        "conformal_margin_width":  float(margin),
        **compute_pi_coverage(y_arr, preds_raw, pi_lower, pi_upper),
        **compute_aqi_event_metrics(y_arr, preds_raw, 150),
        **compute_aqi_event_metrics(y_arr, preds_raw, 200),
        "error_by_band":           error_analysis(y_arr, preds_raw),
        "quantile_errors":         quantile_error_analysis(y_arr, preds_raw),
    }
    if extra:
        metrics.update(extra)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(horizon: int) -> dict:
    print(f"\n{'=' * 70}\n  Random Forest — {horizon}h Horizon\n{'=' * 70}")

    # 1. Load
    X, y_log = load_xy(horizon, use_log=True)
    _, y_raw = load_xy(horizon, use_log=False)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # 2. Correlation filter
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97
    )
    print(f"  Features: {X_train.shape[1]} kept, {len(dropped_cols)} dropped by corr filter.")

    # 3. Spike augmentation (15% target fraction)
    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train, y_train_log, y_train_raw=y_train_raw,
        spike_threshold=150, target_spike_fraction=0.15,
    )

    # 4. Sample weights — exponential emphasis on high-AQI events
    y_aug_raw = np.expm1(y_train_aug.values)
    sample_weights = 1.0 + np.exp(np.minimum(y_aug_raw, 350) / 110.0) - np.exp(0)
    sample_weights = np.clip(sample_weights, 1.0, 25.0)

    # 5. Hyperparameter search (RandomizedSearchCV, TimeSeriesSplit)
    param_dist = {
        "n_estimators":      [300, 500, 700],
        "max_depth":         [15, 20, 28, None],
        "min_samples_leaf":  [2, 3, 5, 8],
        "min_samples_split": [4, 6, 10, 14],
        "max_features":      [0.15, 0.2, 0.3, 0.4],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=15,
        cv=TimeSeriesSplit(n_splits=3, gap=horizon),
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0,
    )
    print("  Hyperparameter search (15 fits) ...")
    search.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)
    model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    best_idx = search.best_index_
    cv_rmse  = float(-search.cv_results_["mean_test_score"][best_idx])
    cv_std   = float( search.cv_results_["std_test_score"][best_idx])

    # 6. Conformal calibration
    cal_preds_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin        = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_preds_raw))

    # 7. Test inference
    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    y_arr     = y_test_raw.values
    pi_lower  = np.clip(preds_raw - margin, 0, 500)
    pi_upper  = np.clip(preds_raw + margin, 0, 500)

    # 8. Persistence baseline skill score
    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - float(mean_absolute_error(y_arr, preds_raw)) / p_mae) if p_mae > 0 else 0.0
        r2_imp = r2_score(y_arr, preds_raw) - p_r2
        print(f"  Persistence baseline: {lag_col}  MAE={p_mae:.1f}  skill={skill:.3f}")

    # 9. SHAP
    top_features, shap_df = run_shap(model, X_test, "RF", horizon, n_sample=150)
    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "RF", top_features)
    except Exception as e:
        print(f"  Drift monitoring skipped: {e}")

    # 10. Metrics
    metrics = build_base_metrics(
        "RandomForest", horizon, y_arr, preds_raw, pi_lower, pi_upper,
        margin, cv_rmse, cv_std,
        extra={
            "test_median_ae":               float(median_absolute_error(y_arr, preds_raw)),
            "best_params":                  search.best_params_,
            "baseline_lag_col":             lag_col or "none",
            "baseline_horizon_mae":         float(p_mae),
            "baseline_horizon_r2":          float(p_r2),
            "forecast_skill_score":         float(skill),
            "r2_improvement_vs_baseline":   float(r2_imp),
        }
    )

    # 11. Persist everything to MongoDB
    artifact = {
        "model":            model,
        "feature_names":    list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log":          True,
    }
    shap_records = shap_df.head(30).to_dict(orient="records") if not shap_df.empty else []
    push_model(horizon, "random_forest", artifact, metrics,
               list(X_train.columns), shap_records)
    push_metrics(horizon, "random_forest", metrics)
    push_predictions(horizon, "random_forest", y_arr, preds_raw,
                     pi_lower, pi_upper, X_test.index)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Ridge Regression
# ══════════════════════════════════════════════════════════════════════════════

def train_ridge(horizon: int) -> dict:
    print(f"\n{'=' * 70}\n  Ridge Regression — {horizon}h Horizon\n{'=' * 70}")

    # 1. Load
    X, y_log = load_xy(horizon, use_log=True)
    _, y_raw = load_xy(horizon, use_log=False)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # 2. Correlation filter
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.95
    )
    print(f"  Features: {X_train.shape[1]} kept, {len(dropped_cols)} dropped.")

    # 3. Imputation (Ridge cannot handle NaN natively)
    X_train, X_cal, X_test = impute_for_linear(X_train, X_cal, X_test)
    print("  Train-mean imputation applied (fitted on X_train only — no leakage).")

    # 4. TimeSeriesSplit CV to pick best alpha
    tscv      = TimeSeriesSplit(n_splits=4, gap=horizon)
    fold_rmse = []
    for tr_idx, val_idx in tscv.split(X_train):
        fold_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  RidgeCV(alphas=np.logspace(-3, 3, 20))),
        ])
        fold_pipe.fit(X_train.iloc[tr_idx], y_train_log.iloc[tr_idx])
        fold_pred_raw = np.expm1(np.clip(fold_pipe.predict(X_train.iloc[val_idx]), 0, None))
        fold_rmse.append(root_mean_squared_error(
            np.expm1(y_train_log.iloc[val_idx].values), fold_pred_raw
        ))
    cv_rmse = float(np.mean(fold_rmse))
    cv_std  = float(np.std(fold_rmse))
    print(f"  CV RMSE: {cv_rmse:.2f} ± {cv_std:.2f}")

    # 5. Final model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TimeSeriesSplit(n_splits=5))),
    ])
    model.fit(X_train, y_train_log)
    best_alpha = float(model.named_steps["ridge"].alpha_)
    print(f"  Optimal alpha: {best_alpha:.4f}")

    # 6. Conformal calibration
    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin       = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    # 7. Test inference
    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    y_arr     = y_test_raw.values
    pi_lower  = np.clip(preds_raw - margin, 0, 500)
    pi_upper  = np.clip(preds_raw + margin, 0, 500)

    # 8. Persistence baseline
    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - float(mean_absolute_error(y_arr, preds_raw)) / p_mae) if p_mae > 0 else 0.0
        r2_imp = r2_score(y_arr, preds_raw) - p_r2

    # 9. Coefficient importance (Ridge equivalent of feature importance)
    coef_df = pd.DataFrame({
        "feature":  X_train.columns,
        "abs_coef": np.abs(model.named_steps["ridge"].coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    top_features  = coef_df.head(20)["feature"].tolist()
    coef_records  = coef_df.head(30).to_dict(orient="records")

    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "Ridge", top_features)
    except Exception as e:
        print(f"  Drift monitoring skipped: {e}")

    # 10. Metrics
    metrics = build_base_metrics(
        "Ridge", horizon, y_arr, preds_raw, pi_lower, pi_upper,
        margin, cv_rmse, cv_std,
        extra={
            "best_alpha":                   best_alpha,
            "baseline_lag_col":             lag_col or "none",
            "baseline_horizon_mae":         float(p_mae),
            "baseline_horizon_r2":          float(p_r2),
            "forecast_skill_score":         float(skill),
            "r2_improvement_vs_baseline":   float(r2_imp),
        }
    )

    # 11. Persist to MongoDB
    artifact = {
        "model":            model,
        "feature_names":    list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log":          True,
    }
    push_model(horizon, "ridge", artifact, metrics,
               list(X_train.columns), coef_records)
    push_metrics(horizon, "ridge", metrics)
    push_predictions(horizon, "ridge", y_arr, preds_raw,
                     pi_lower, pi_upper, X_test.index)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(horizon: int) -> dict:
    if not XGB_AVAILABLE:
        print(f"  XGBoost not available — skipping horizon {horizon}h.")
        return {}

    print(f"\n{'=' * 70}\n  XGBoost — {horizon}h Horizon\n{'=' * 70}")

    # 1. Load
    X, y_log = load_xy(horizon, use_log=True)
    _, y_raw = load_xy(horizon, use_log=False)

    X_train, y_train_log, X_cal, y_cal_log, X_test, y_test_log = \
        get_chronological_splits(X, y_log, horizon)
    _, y_train_raw, _, y_cal_raw, _, y_test_raw = \
        get_chronological_splits(X, y_raw, horizon)

    # 2. Correlation filter
    X_train, X_cal, X_test, dropped_cols = apply_leakage_free_correlation_filter(
        X_train, X_test, X_cal, threshold=0.97
    )
    print(f"  Features: {X_train.shape[1]} kept, {len(dropped_cols)} dropped.")

    # 3. Spike augmentation (7% target — matches Karachi's ~8.6% spike rate)
    X_train_aug, y_train_aug = get_spike_augmented_train(
        X_train, y_train_log, y_train_raw=y_train_raw,
        spike_threshold=150, target_spike_fraction=0.07,
    )

    # 4. Sample weights
    y_aug_raw       = np.expm1(y_train_aug.values)
    sample_weights  = np.ones(len(y_train_aug))
    sample_weights[y_aug_raw > 100] = 2.0
    sample_weights[y_aug_raw > 150] = 4.0
    sample_weights[y_aug_raw > 200] = 8.0

    # 5. TimeSeriesSplit CV (CV folds use matching weights)
    tscv      = TimeSeriesSplit(n_splits=5, gap=horizon)
    fold_rmse = []
    print("  5-fold TimeSeriesCV ...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_ft, y_ft = X_train.iloc[tr_idx], y_train_log.iloc[tr_idx]
        X_fv, y_fv = X_train.iloc[val_idx], y_train_log.iloc[val_idx]

        fw = np.ones(len(y_ft))
        fw[np.expm1(y_ft.values) > 100] = 2.0
        fw[np.expm1(y_ft.values) > 150] = 4.0
        fw[np.expm1(y_ft.values) > 200] = 8.0

        fm = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, random_state=42 + fold, n_jobs=-1, verbosity=0,
        )
        fm.fit(X_ft, y_ft, sample_weight=fw)
        fold_pred_raw = np.expm1(np.clip(fm.predict(X_fv), 0, None))
        fold_rmse.append(root_mean_squared_error(
            np.expm1(y_fv.values), fold_pred_raw
        ))
    cv_rmse = float(np.mean(fold_rmse))
    cv_std  = float(np.std(fold_rmse))
    print(f"  CV RMSE: {cv_rmse:.2f} ± {cv_std:.2f}")

    # 6. Final model with early stopping on calibration set
    model = xgb.XGBRegressor(
        n_estimators=1200, max_depth=6, learning_rate=0.015,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, tree_method="hist",
        early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    print("  Training final model (early stopping on cal set) ...")
    model.fit(
        X_train_aug, y_train_aug,
        sample_weight=sample_weights,
        eval_set=[(X_cal, y_cal_log)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # 7. Conformal calibration
    cal_pred_raw = np.expm1(np.clip(model.predict(X_cal), 0, None))
    margin       = calculate_conformal_margin(np.abs(y_cal_raw.values - cal_pred_raw))

    # 8. Test inference
    preds_raw = np.clip(np.expm1(model.predict(X_test)), 0, 500)
    y_arr     = y_test_raw.values
    pi_lower  = np.clip(preds_raw - margin, 0, 500)
    pi_upper  = np.clip(preds_raw + margin, 0, 500)

    # 9. Persistence baseline
    lag_col = get_persistence_baseline_col(X_test, horizon)
    p_mae = p_r2 = skill = r2_imp = 0.0
    if lag_col:
        p_mae  = mean_absolute_error(y_arr, X_test[lag_col].values)
        p_r2   = r2_score(y_arr, X_test[lag_col].values)
        skill  = float(1.0 - float(mean_absolute_error(y_arr, preds_raw)) / p_mae) if p_mae > 0 else 0.0
        r2_imp = r2_score(y_arr, preds_raw) - p_r2
        print(f"  Persistence baseline: {lag_col}  MAE={p_mae:.1f}  skill={skill:.3f}")

    # 10. SHAP
    top_features, shap_df = run_shap(model, X_test, "XGB", horizon, n_sample=300)
    try:
        run_data_drift_monitoring(X_train, X_test, horizon, "XGB", top_features)
    except Exception as e:
        print(f"  Drift monitoring skipped: {e}")

    # 11. Metrics
    metrics = build_base_metrics(
        "XGBoost", horizon, y_arr, preds_raw, pi_lower, pi_upper,
        margin, cv_rmse, cv_std,
        extra={
            "best_iteration":               model.best_iteration,
            "baseline_lag_col":             lag_col or "none",
            "baseline_horizon_mae":         float(p_mae),
            "baseline_horizon_r2":          float(p_r2),
            "forecast_skill_score":         float(skill),
            "r2_improvement_vs_baseline":   float(r2_imp),
        }
    )

    # 12. Persist to MongoDB
    artifact = {
        "model":            model,
        "feature_names":    list(X_train.columns),
        "conformal_margin": float(margin),
        "use_log":          True,
    }
    shap_records = shap_df.head(30).to_dict(orient="records") if not shap_df.empty else []
    push_model(horizon, "xgboost", artifact, metrics,
               list(X_train.columns), shap_records)
    push_metrics(horizon, "xgboost", metrics)
    push_predictions(horizon, "xgboost", y_arr, preds_raw,
                     pi_lower, pi_upper, X_test.index)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Evaluation Summary  (replaces evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation_summary(all_results: dict):
    """
    Prints the cross-model comparison table, flags the best model per horizon,
    and pushes a pipeline_run summary document to MongoDB.
    """
    MODEL_NAMES = ["random_forest", "ridge", "xgboost"]
    DISPLAY     = {"random_forest": "Random Forest", "ridge": "Ridge", "xgboost": "XGBoost"}

    print("\n" + "=" * 115)
    print("  CROSS-MODEL EVALUATION SUMMARY")
    print("=" * 115)
    print(f"  {'Horizon':<8} {'Model':<16} {'Test MAE':>10} {'MAPE':>9} "
          f"{'R²':>9} {'Coverage':>11} {'Cov>150':>9} {'Cov>200':>9} {'Skill':>8}")
    print("-" * 115)

    run_summary = {"horizons": {}}

    for h in HORIZONS:
        horizon_key   = f"{h}h"
        best_r2       = -np.inf
        best_name     = None
        horizon_block = {}

        for name in MODEL_NAMES:
            m = all_results.get(horizon_key, {}).get(name, {})
            if not m:
                continue

            r2       = m.get("test_r2",    0.0)
            mae      = m.get("test_mae",   0.0)
            mape     = m.get("test_mape",  0.0)
            cov      = m.get("conformal_global_coverage", 0.0)
            cov_150  = m.get("conformal_coverage_gt150",  0.0)
            cov_200  = m.get("conformal_coverage_gt200",  0.0)
            skill    = m.get("forecast_skill_score",      0.0)

            print(f"  {horizon_key:<8} {DISPLAY[name]:<16} "
                  f"{mae:>10.1f} {mape:>8.2f}% {r2:>9.3f} "
                  f"{cov*100:>10.1f}% {cov_150*100:>8.1f}% {cov_200*100:>8.1f}% {skill:>8.3f}")

            horizon_block[name] = {"r2": r2, "mae": mae}
            if r2 > best_r2:
                best_r2, best_name = r2, name

        if best_name:
            flag_best_model(h, best_name)
            print(f"  {'':8} ★ Best: {DISPLAY[best_name]}  (R²={best_r2:.4f})")
            run_summary["horizons"][horizon_key] = {
                "best_model": best_name,
                "best_r2":    round(best_r2, 4),
                "models":     horizon_block,
            }
        print()

    print("=" * 115)
    push_pipeline_run_summary(run_summary)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  KARACHI AQI — CONSOLIDATED TRAINING PIPELINE")
    print("=" * 70)

    all_results: dict[str, dict] = {}

    for horizon in HORIZONS:
        hk = f"{horizon}h"
        all_results[hk] = {}

        m_rf    = train_random_forest(horizon)
        all_results[hk]["random_forest"] = m_rf

        m_ridge = train_ridge(horizon)
        all_results[hk]["ridge"] = m_ridge

        if XGB_AVAILABLE:
            m_xgb = train_xgboost(horizon)
            all_results[hk]["xgboost"] = m_xgb

    run_evaluation_summary(all_results)

    print("\nAll models, metrics, and predictions have been pushed to MongoDB.")
    print("No local files were written. The pipeline is fully serverless.")


if __name__ == "__main__":
    main()