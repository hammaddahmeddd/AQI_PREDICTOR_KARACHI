from fastapi import APIRouter, HTTPException, Query
from pymongo import DESCENDING, ASCENDING
import gridfs
import io
import joblib
import numpy as np
import pandas as pd

from app.db import get_db
from app.utils import clean_doc, aqi_category, safe_round, model_display_name, HORIZONS

router = APIRouter()

def get_latest_feature_doc():
    db = get_db()
    doc = db["processed_features"].find_one({}, {"_id": 0}, sort=[("datetime", DESCENDING)])
    if not doc:
        doc = db["karachi_aqi_dataset"].find_one({}, {"_id": 0}, sort=[("datetime", DESCENDING)])
    return doc

def get_latest_metrics_map():
    db = get_db()
    rows = list(db["model_metrics"].aggregate([
        {"$match": {"horizon": {"$in": HORIZONS}, "model_name": {"$exists": True}}},
        {"$sort": {"logged_at": -1}},
        {"$group": {
            "_id": {"horizon": "$horizon", "model_name": "$model_name"},
            "doc": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$doc"}},
        {"$project": {"_id": 0}}
    ]))
    metrics = {}
    for row in rows:
        metrics.setdefault(str(row.get("horizon")), {})[row.get("model_name")] = row
    return metrics

def load_artifact(horizon: int, model_name: str):
    db = get_db()
    registry_query = {"horizon": horizon, "model_name": model_name}
    registry_doc = db["model_registry"].find_one(registry_query, sort=[("trained_at", DESCENDING)])
    if not registry_doc:
        raise HTTPException(status_code=404, detail=f"No model found for {model_name} {horizon}h.")

    if registry_doc.get("artifact_storage") == "gridfs":
        fs = gridfs.GridFS(db, collection="model_artifacts")
        artifact_file_id = registry_doc.get("artifact_gridfs_id")
        if not artifact_file_id:
            raise HTTPException(status_code=404, detail="Registry document missing artifact_gridfs_id.")
        grid_out = fs.get(artifact_file_id)
        artifact = joblib.load(io.BytesIO(grid_out.read()))
    elif "model_binary" in registry_doc:
        artifact = joblib.load(io.BytesIO(registry_doc["model_binary"]))
    else:
        raise HTTPException(status_code=404, detail="No model artifact found in registry document.")

    return artifact, registry_doc

@router.get("/health")
def health():
    db = get_db()
    db.command("ping")
    return {"status": "ok"}

@router.get("/summary")
def summary():
    db = get_db()
    feature_count = db["processed_features"].count_documents({})
    latest = clean_doc(get_latest_feature_doc())

    pipeline_status = list(db["pipeline_status"].find(
        {}, {"_id": 0}, sort=[("logged_at", DESCENDING)], limit=8
    ))

    latest_run = db["pipeline_runs"].find_one({}, {"_id": 0}, sort=[("run_at", DESCENDING)])
    return {
        "feature_count": feature_count,
        "latest": latest,
        "latest_run": clean_doc(latest_run),
        "pipeline_status": [clean_doc(item) for item in pipeline_status],
    }

@router.get("/current")
def current():
    doc = clean_doc(get_latest_feature_doc())
    if not doc:
        raise HTTPException(status_code=404, detail="No current AQI document found.")

    aqi = doc.get("aqi")
    if aqi is None and doc.get("pm25") is not None:
        pm25 = float(doc["pm25"])
        if pm25 <= 12:
            aqi = round((50 / 12) * pm25)
        elif pm25 <= 35.4:
            aqi = round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
        elif pm25 <= 55.4:
            aqi = round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
        else:
            aqi = min(round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151), 500)

    return {
        "datetime": doc.get("datetime") or doc.get("timestamp"),
        "aqi": safe_round(aqi, 0),
        "category": aqi_category(aqi),
        "pollutants": {
            "pm25": safe_round(doc.get("pm25"), 1),
            "pm10": safe_round(doc.get("pm10"), 1),
            "no2": safe_round(doc.get("no2"), 1),
            "so2": safe_round(doc.get("so2"), 1),
            "co": safe_round(doc.get("co"), 2),
            "o3": safe_round(doc.get("o3"), 1),
        },
        "weather": {
            "temperature": safe_round(doc.get("temperature"), 1),
            "humidity": safe_round(doc.get("humidity"), 1),
            "pressure": safe_round(doc.get("pressure"), 0),
            "wind_speed": safe_round(doc.get("wind_speed"), 1),
            "wind_direction": safe_round(doc.get("wind_direction"), 1),
            "precipitation": safe_round(doc.get("precipitation"), 2),
        }
    }

@router.get("/metrics")
def metrics():
    db = get_db()

    latest_rows = list(db["model_metrics"].aggregate([
        {"$match": {"horizon": {"$in": HORIZONS}, "model_name": {"$exists": True}}},
        {"$sort": {"logged_at": -1}},
        {"$group": {
            "_id": {"horizon": "$horizon", "model_name": "$model_name"},
            "doc": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$doc"}},
        {"$project": {"_id": 0}}
    ]))

    registry_rows = list(db["model_registry"].find(
        {"horizon": {"$in": HORIZONS}},
        {"_id": 0, "horizon": 1, "model_name": 1, "is_best": 1, "trained_at": 1, "n_features": 1, "top_features": 1}
    ))

    best_lookup = {
        (row.get("horizon"), row.get("model_name")): row
        for row in registry_rows
    }

    output = []
    for row in latest_rows:
        reg = best_lookup.get((row.get("horizon"), row.get("model_name")), {})
        output.append({
            "horizon": row.get("horizon"),
            "model_name": row.get("model_name"),
            "model_label": model_display_name(row.get("model_name", "")),
            "is_best": bool(reg.get("is_best", False)),
            "trained_at": row.get("logged_at") or reg.get("trained_at"),
            "test_mae": safe_round(row.get("test_mae"), 2),
            "test_rmse": safe_round(row.get("test_rmse"), 2),
            "test_r2": safe_round(row.get("test_r2"), 4),
            "test_mape": safe_round(row.get("test_mape"), 2),
            "coverage": safe_round((row.get("conformal_global_coverage") or 0) * 100, 1),
            "coverage_gt150": safe_round((row.get("conformal_coverage_gt150") or 0) * 100, 1),
            "coverage_gt200": safe_round((row.get("conformal_coverage_gt200") or 0) * 100, 1),
            "skill": safe_round(row.get("forecast_skill_score"), 4),
            "n_features": reg.get("n_features"),
            "top_features": reg.get("top_features", [])[:8],
        })

    output.sort(key=lambda x: (x["horizon"], x["model_name"]))
    return {"items": output}

@router.get("/history")
def history(limit: int = Query(48, ge=6, le=336)):
    db = get_db()
    docs = list(db["processed_features"].find(
        {}, {"_id": 0, "datetime": 1, "timestamp": 1, "aqi": 1, "pm25": 1, "pm10": 1, "temperature": 1, "humidity": 1, "wind_speed": 1},
        sort=[("datetime", DESCENDING)],
        limit=limit
    ))
    docs = list(reversed([clean_doc(doc) for doc in docs]))
    return {"items": docs}

@router.get("/predictions")
def predictions(
    horizon: int = Query(24, enum=HORIZONS),
    model: str = Query("xgboost")
):
    db = get_db()
    rows = list(db["model_predictions"].find(
        {"horizon": horizon, "model_name": model},
        {"_id": 0},
        sort=[("row_index", ASCENDING)],
        limit=250
    ))
    return {"items": [clean_doc(row) for row in rows]}

@router.post("/predict")
def predict(payload: dict):
    horizon = int(payload.get("horizon", 24))
    model_name = payload.get("model_name", "xgboost")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="horizon must be 24, 48, or 72.")

    artifact, registry_doc = load_artifact(horizon, model_name)
    model = artifact["model"]
    feature_names = artifact.get("feature_names") or registry_doc.get("feature_names")
    if not feature_names:
        raise HTTPException(status_code=400, detail="Model artifact has no feature_names.")

    latest = get_latest_feature_doc()
    if not latest:
        raise HTTPException(status_code=404, detail="No latest feature row available.")

    feature_row = {}
    for name in feature_names:
        value = latest.get(name)
        feature_row[name] = np.nan if value is None else value

    X = pd.DataFrame([feature_row], columns=feature_names)
    X = X.replace([np.inf, -np.inf], np.nan)

    raw_pred = model.predict(X)[0]
    pred = np.expm1(raw_pred) if artifact.get("use_log", True) else raw_pred
    pred = float(np.clip(pred, 0, 500))

    margin = float(artifact.get("conformal_margin", 0))
    lower = float(np.clip(pred - margin, 0, 500))
    upper = float(np.clip(pred + margin, 0, 500))

    return {
        "horizon": horizon,
        "model_name": model_name,
        "model_label": model_display_name(model_name),
        "prediction": round(pred, 1),
        "pi_lower": round(lower, 1),
        "pi_upper": round(upper, 1),
        "category": aqi_category(pred),
    }
