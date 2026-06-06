"""
main.py — FastAPI Backend for Karachi AQI Dashboard
Deployed on Render (free tier). All data comes from MongoDB.

Endpoints:
  GET  /                        → health check
  GET  /api/metrics             → all model metrics for all horizons
  GET  /api/metrics/{horizon}   → metrics for a specific horizon (24, 48, 72)
  GET  /api/predictions/{model}/{horizon} → latest predictions
  GET  /api/latest-aqi          → most recent AQI reading from feature store
  GET  /api/model-comparison    → cross-model comparison table
  GET  /api/top-features/{model}/{horizon} → SHAP top features
  GET  /api/pipeline-status     → recent pipeline run statuses
"""

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Karachi AQI Forecast API",
    description="Real-time AQI forecasting for Karachi using ML models trained on Open-Meteo data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── MongoDB connection ─────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "karachi_aqi"

VALID_HORIZONS = {24, 48, 72}
VALID_MODELS = {"random_forest", "ridge", "xgboost"}
MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
}


def get_db():
    if not MONGO_URI:
        raise HTTPException(status_code=500, detail="MONGODB_URI not configured on server.")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        return client[DB_NAME], client
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e}")


def _clean_doc(doc: dict) -> dict:
    """Remove MongoDB _id and convert datetimes to ISO strings."""
    doc.pop("_id", None)
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Karachi AQI Forecast API",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_all_metrics():
    """
    Returns the latest metrics for all models across all horizons.
    Shape: { "24h": { "random_forest": {...}, "ridge": {...}, "xgboost": {...} }, ... }
    """
    db, client = get_db()
    try:
        result = {}
        for horizon in VALID_HORIZONS:
            hk = f"{horizon}h"
            result[hk] = {}
            for model in VALID_MODELS:
                doc = db["model_metrics"].find_one(
                    {"horizon": horizon, "model_name": model},
                    {"_id": 0, "artifact_gridfs_id": 0, "artifact_filename": 0,
                     "artifact_storage": 0, "feature_names": 0},
                    sort=[("trained_at", DESCENDING)],
                )
                if doc:
                    result[hk][model] = _clean_doc(doc)
        return result
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


@app.get("/api/metrics/{horizon}")
def get_metrics_by_horizon(horizon: int):
    """Returns latest metrics for all models at a specific horizon."""
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {VALID_HORIZONS}")

    db, client = get_db()
    try:
        result = {}
        for model in VALID_MODELS:
            doc = db["model_metrics"].find_one(
                {"horizon": horizon, "model_name": model},
                {"_id": 0, "artifact_gridfs_id": 0, "artifact_filename": 0,
                 "artifact_storage": 0, "feature_names": 0},
                sort=[("trained_at", DESCENDING)],
            )
            if doc:
                result[model] = _clean_doc(doc)
        return {"horizon": horizon, "models": result}
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


# ── Model comparison ──────────────────────────────────────────────────────────

@app.get("/api/model-comparison")
def get_model_comparison():
    """
    Returns a flat list of comparison rows suitable for building a table.
    Each row: { horizon, model, display_name, mae, mape, r2, coverage,
                coverage_gt150, coverage_gt200, skill, trained_at }
    """
    db, client = get_db()
    try:
        rows = []
        for horizon in sorted(VALID_HORIZONS):
            for model in VALID_MODELS:
                doc = db["model_metrics"].find_one(
                    {"horizon": horizon, "model_name": model},
                    {"_id": 0},
                    sort=[("trained_at", DESCENDING)],
                )
                if not doc:
                    continue
                rows.append({
                    "horizon": horizon,
                    "horizon_label": f"{horizon}h",
                    "model": model,
                    "display_name": MODEL_DISPLAY[model],
                    "mae":           _safe_float(doc, "test_mae"),
                    "rmse":          _safe_float(doc, "test_rmse"),
                    "mape":          _safe_float(doc, "test_mape"),
                    "r2":            _safe_float(doc, "test_r2"),
                    "median_ae":     _safe_float(doc, "test_median_ae"),
                    "coverage":      _safe_float(doc, "conformal_global_coverage"),
                    "coverage_gt150": _safe_float(doc, "conformal_coverage_gt150"),
                    "coverage_gt200": _safe_float(doc, "conformal_coverage_gt200"),
                    "skill":         _safe_float(doc, "forecast_skill_score"),
                    "cv_rmse":       _safe_float(doc, "cv_rmse"),
                    "trained_at":    doc.get("trained_at", ""),
                })
        return {"rows": rows, "count": len(rows)}
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


def _safe_float(doc: dict, key: str) -> Optional[float]:
    v = doc.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Predictions ───────────────────────────────────────────────────────────────

@app.get("/api/predictions/{model}/{horizon}")
def get_predictions(model: str, horizon: int, limit: int = Query(default=168, ge=1, le=2000)):
    """
    Returns the most recent `limit` prediction rows for a model/horizon combo.
    Each row: { timestamp, actual, predicted, pi_lower, pi_upper }
    """
    if model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Model must be one of {list(VALID_MODELS)}")
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {VALID_HORIZONS}")

    db, client = get_db()
    try:
        collection_name = f"predictions_{model}_{horizon}h"
        # Sort by row_index DESCENDING to get the most recent rows, then reverse
        # for chronological order. row_index is the positional index from the
        # test split — higher = later in time.
        docs = list(
            db[collection_name]
            .find({}, {"_id": 0})
            .sort("row_index", DESCENDING)
            .limit(limit)
        )
        docs.reverse()  # chronological order for the chart
        cleaned = [_clean_doc(d) for d in docs]
        return {
            "model": model,
            "display_name": MODEL_DISPLAY[model],
            "horizon": horizon,
            "count": len(cleaned),
            "predictions": cleaned,
        }
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


# ── Latest AQI ────────────────────────────────────────────────────────────────

@app.get("/api/latest-aqi")
def get_latest_aqi():
    """
    Returns the most recent AQI reading and key pollutants from the feature store.
    """
    db, client = get_db()
    try:
        doc = db["processed_features"].find_one(
            {},
            {"_id": 0, "timestamp": 1, "datetime": 1, "aqi": 1,
             "pm25": 1, "pm10": 1, "co": 1, "no2": 1, "so2": 1, "o3": 1,
             "temperature": 1, "humidity": 1, "wind_speed": 1,
             "aqi_lag_1": 1, "aqi_lag_6": 1, "aqi_lag_24": 1},
            sort=[("timestamp", DESCENDING)],
        )
        if not doc:
            raise HTTPException(status_code=404, detail="No AQI data found in feature store.")
        return _clean_doc(doc)
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


# ── Recent AQI history ────────────────────────────────────────────────────────

@app.get("/api/aqi-history")
def get_aqi_history(hours: int = Query(default=72, ge=1, le=720)):
    """Returns the last N hours of AQI readings from the feature store."""
    db, client = get_db()
    try:
        docs = list(
            db["processed_features"]
            .find(
                {},
                {"_id": 0, "timestamp": 1, "datetime": 1, "aqi": 1,
                 "pm25": 1, "pm10": 1, "temperature": 1, "humidity": 1,
                 "wind_speed": 1, "precipitation": 1},
            )
            .sort("timestamp", DESCENDING)
            .limit(hours)
        )
        docs.reverse()
        return {"hours": hours, "count": len(docs), "history": [_clean_doc(d) for d in docs]}
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


# ── Top features (SHAP) ───────────────────────────────────────────────────────

@app.get("/api/top-features/{model}/{horizon}")
def get_top_features(model: str, horizon: int):
    """Returns SHAP top features from the model registry."""
    if model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Model must be one of {list(VALID_MODELS)}")
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {VALID_HORIZONS}")

    db, client = get_db()
    try:
        doc = db["model_registry"].find_one(
            {"horizon": horizon, "model_name": model},
            {"_id": 0, "top_features": 1, "trained_at": 1, "n_features": 1},
            sort=[("trained_at", DESCENDING)],
        )
        if not doc:
            raise HTTPException(status_code=404, detail="No registry entry found for this model/horizon.")
        return {
            "model": model,
            "display_name": MODEL_DISPLAY[model],
            "horizon": horizon,
            "n_features": doc.get("n_features"),
            "trained_at": doc.get("trained_at"),
            "top_features": doc.get("top_features", []),
        }
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()


# ── Pipeline status ───────────────────────────────────────────────────────────

@app.get("/api/pipeline-status")
def get_pipeline_status(limit: int = Query(default=20, ge=1, le=100)):
    """Returns recent pipeline run statuses from both feature and training pipelines."""
    db, client = get_db()
    try:
        docs = list(
            db["pipeline_status"]
            .find({}, {"_id": 0})
            .sort("logged_at", DESCENDING)
            .limit(limit)
        )
        return {"count": len(docs), "statuses": [_clean_doc(d) for d in docs]}
    except PyMongoError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        client.close()