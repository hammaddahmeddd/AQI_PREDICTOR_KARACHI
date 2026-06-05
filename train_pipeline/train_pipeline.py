"""
train_pipeline.py
-----------------
Consolidated training pipeline for Karachi AQI forecasting.
"""

import os
import sys
import io
import joblib
import numpy as np
import pandas as pd
import pymongo
from pathlib import Path
from bson.binary import Binary
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensures the root directory is in sys.path so we can import 'load_data.py'
# even though this script is sitting inside the 'train_pipeline' subfolder.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Now import from the root directory ────────────────────────────────────────
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

# ── Machine Learning Imports ──────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
HORIZONS = [24, 48, 72]
DISPLAY = {
    "random_forest": "Random Forest",
    "ridge":         "Ridge Regression",
    "xgboost":       "XGBoost"
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MongoDB helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI not set")
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    return client["karachi_aqi"]

def save_model_to_mongodb(horizon, model_name, model_obj, metadata):
    db = get_db()
    col = db["model_registry"]
    
    # Serialise model to bytes
    buffer = io.BytesIO()
    joblib.dump(model_obj, buffer)
    
    doc = {
        "horizon": horizon,
        "model_name": model_name,
        "model_binary": Binary(buffer.getvalue()),
        "metadata": metadata,
        "trained_at": datetime.now(tz=timezone.utc),
        "is_best": False  # Updated later by flag_best_model
    }
    
    col.update_one(
        {"horizon": horizon, "model_name": model_name},
        {"$set": doc},
        upsert=True
    )

def flag_best_model(horizon, model_name):
    db = get_db()
    col = db["model_registry"]
    col.update_many({"horizon": horizon}, {"$set": {"is_best": False}})
    col.update_one({"horizon": horizon, "model_name": model_name}, {"$set": {"is_best": True}})

def push_pipeline_run_summary(summary_dict):
    db = get_db()
    summary_dict["completed_at"] = datetime.now(tz=timezone.utc)
    db["pipeline_runs"].insert_one(summary_dict)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Training Wrapper (Example for RF)
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(horizon):
    print(f"  [Random Forest] Training for {horizon}h horizon...")
    X, y = load_xy(horizon, use_log=True)
    X_train, y_train, X_cal, y_cal, X_test, y_test = get_chronological_splits(X, y, horizon)
    
    # Fillna for RF (Trees handle NaN but sklearn RF requires finite values)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics = {"MAE": float(mae), "R2": float(r2)}
    save_model_to_mongodb(horizon, "random_forest", model, metrics)
    return metrics

# ... (Note: You would include train_ridge and train_xgboost functions here similarly)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  KARACHI AQI — CONSOLIDATED TRAINING PIPELINE")
    print("=" * 70)

    run_summary = {
        "run_id": datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "horizons": {}
    }

    for h in HORIZONS:
        horizon_key = f"{h}h"
        print(f"\n>>> Processing Horizon: {horizon_key}")
        
        # Train and collect results
        rf_metrics = train_random_forest(h)
        # (Assuming you add the other training functions to the script)
        
        # For this example, we just evaluate the one we have
        best_name = "random_forest"
        best_r2 = rf_metrics["R2"]
        
        flag_best_model(h, best_name)
        print(f"  ★ Best: {DISPLAY[best_name]} (R²={best_r2:.4f})")
        
        run_summary["horizons"][horizon_key] = {
            "best_model": best_name,
            "best_r2": round(best_r2, 4)
        }

    print("\n" + "=" * 70)
    push_pipeline_run_summary(run_summary)
    print("  TRAINING COMPLETE — Results pushed to MongoDB.")

if __name__ == "__main__":
    main()