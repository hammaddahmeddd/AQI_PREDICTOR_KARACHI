"""
run_feature_pipeline.py
------------------------
Orchestrates the full hourly feature pipeline:
  1. Fetch fresh weather data   → MongoDB raw_weather
  2. Fetch fresh air quality    → MongoDB raw_air_quality
  3. Merge raw collections      → MongoDB karachi_aqi_dataset
  4. Build engineered features  → MongoDB processed_features

Run manually:
    MONGODB_URI="..." python run_feature_pipeline.py

Run via GitHub Actions: .github/workflows/feature_pipeline.yml (every hour)
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pymongo
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "feature_pipeline"))
sys.path.insert(0, str(BASE_DIR / "training_pipeline"))

from data_pipeline.fetch_weather import fetch_weather, save_to_mongodb as save_weather
from data_pipeline.fetch_air_quality import fetch_air_quality, save_to_mongodb as save_aq
from data_pipeline.build_dataset import build_and_save_dataset
from training_pipeline.feature_engineering import process_all as run_feature_engineering


def log(msg: str):
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def push_pipeline_status(db, step: str, status: str, error: str = ""):
    """Record each step's outcome for monitoring in MongoDB."""
    try:
        db["pipeline_status"].insert_one({
            "pipeline":   "feature",
            "step":       step,
            "status":     status,
            "error":      error,
            "logged_at":  datetime.now(tz=timezone.utc).isoformat(),
        })
    except Exception:
        pass  # Don't let monitoring failures break the pipeline


def main():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("CRITICAL: MONGODB_URI environment variable is not set.")
        sys.exit(1)

    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    db = client["karachi_aqi"]

    log("=" * 70)
    log("  KARACHI AQI — HOURLY FEATURE PIPELINE")
    log("=" * 70)

    overall_ok = True

    # ── STEP 1: Fetch Weather ────────────────────────────────────────────────
    log("STEP 1/4 — Fetching weather data from Open-Meteo...")
    try:
        weather_df = fetch_weather()
        save_weather(weather_df)
        log(f"  ✓ Weather: {len(weather_df):,} rows upserted to raw_weather")
        push_pipeline_status(db, "fetch_weather", "SUCCESS")
    except Exception as e:
        log(f"  ✗ Weather fetch FAILED: {e}")
        traceback.print_exc()
        push_pipeline_status(db, "fetch_weather", "FAILED", str(e))
        overall_ok = False

    # ── STEP 2: Fetch Air Quality ────────────────────────────────────────────
    log("STEP 2/4 — Fetching air quality data from Open-Meteo...")
    try:
        aq_df = fetch_air_quality()
        save_aq(aq_df)
        log(f"  ✓ Air quality: {len(aq_df):,} rows upserted to raw_air_quality")
        push_pipeline_status(db, "fetch_air_quality", "SUCCESS")
    except Exception as e:
        log(f"  ✗ Air quality fetch FAILED: {e}")
        traceback.print_exc()
        push_pipeline_status(db, "fetch_air_quality", "FAILED", str(e))
        overall_ok = False

    # Abort if both fetches failed — nothing to merge
    if not overall_ok:
        log("ABORT: Both data fetches failed. Cannot proceed to merge/feature steps.")
        client.close()
        sys.exit(1)

    # ── STEP 3: Merge into dataset ───────────────────────────────────────────
    log("STEP 3/4 — Merging raw collections → karachi_aqi_dataset...")
    try:
        build_and_save_dataset(db)
        count = db["karachi_aqi_dataset"].count_documents({})
        log(f"  ✓ Dataset: {count:,} merged rows in karachi_aqi_dataset")
        push_pipeline_status(db, "build_dataset", "SUCCESS")
    except Exception as e:
        log(f"  ✗ Dataset build FAILED: {e}")
        traceback.print_exc()
        push_pipeline_status(db, "build_dataset", "FAILED", str(e))
        client.close()
        sys.exit(1)

    # ── STEP 4: Feature Engineering ──────────────────────────────────────────
    log("STEP 4/4 — Running feature engineering → processed_features...")
    try:
        run_feature_engineering()
        count = db["processed_features"].count_documents({})
        log(f"  ✓ Feature store: {count:,} documents in processed_features")
        push_pipeline_status(db, "feature_engineering", "SUCCESS")
    except Exception as e:
        log(f"  ✗ Feature engineering FAILED: {e}")
        traceback.print_exc()
        push_pipeline_status(db, "feature_engineering", "FAILED", str(e))
        client.close()
        sys.exit(1)

    client.close()
    log("=" * 70)
    log("  FEATURE PIPELINE COMPLETE — All 4 steps succeeded.")
    log("=" * 70)


if __name__ == "__main__":
    main()