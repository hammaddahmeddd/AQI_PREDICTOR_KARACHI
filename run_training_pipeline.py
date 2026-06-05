"""
run_training_pipeline.py
-------------------------
Runs the consolidated model training pipeline daily:
  1. Verifies processed_features is populated (feature pipeline must run first)
  2. Trains Random Forest, Ridge, and XGBoost for 24h / 48h / 72h horizons
  3. Pushes all models, metrics, predictions to MongoDB
  4. Flags the best model per horizon
  5. Records a pipeline_runs summary document

Run manually:
    MONGODB_URI="..." python run_training_pipeline.py

Run via GitHub Actions: .github/workflows/training_pipeline.yml (daily at 03:00 UTC)
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pymongo

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "training_pipeline"))

from training_pipeline.train_pipeline import main as run_training


def log(msg: str):
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def verify_feature_store(db) -> int:
    """Returns the number of documents in processed_features."""
    try:
        return db["processed_features"].count_documents({})
    except Exception as e:
        log(f"  WARNING: Could not count processed_features: {e}")
        return 0


def push_run_status(db, status: str, error: str = ""):
    try:
        db["pipeline_status"].insert_one({
            "pipeline":  "training",
            "status":    status,
            "error":     error,
            "logged_at": datetime.now(tz=timezone.utc).isoformat(),
        })
    except Exception:
        pass


def main():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("CRITICAL: MONGODB_URI environment variable is not set.")
        sys.exit(1)

    log("=" * 70)
    log("  KARACHI AQI — DAILY TRAINING PIPELINE")
    log("=" * 70)

    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    db = client["karachi_aqi"]

    # ── Pre-flight check ─────────────────────────────────────────────────────
    log("PRE-FLIGHT — Verifying feature store is populated...")
    n_features = verify_feature_store(db)

    if n_features == 0:
        msg = ("processed_features collection is empty. "
               "The hourly feature pipeline must run successfully before training.")
        log(f"  ✗ ABORT: {msg}")
        push_run_status(db, "ABORTED", msg)
        client.close()
        sys.exit(1)

    log(f"  ✓ Feature store: {n_features:,} documents available for training.")
    client.close()

    # ── Run training ─────────────────────────────────────────────────────────
    log("Starting consolidated training pipeline...")
    try:
        run_training()
        log("=" * 70)
        log("  TRAINING PIPELINE COMPLETE.")
        log("  All models, metrics, and predictions stored in MongoDB.")
        log("=" * 70)

        # Re-open to record final status
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
        push_run_status(client["karachi_aqi"], "SUCCESS")
        client.close()

    except Exception as e:
        log(f"  ✗ Training pipeline FAILED: {e}")
        traceback.print_exc()
        try:
            client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
            push_run_status(client["karachi_aqi"], "FAILED", str(e))
            client.close()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()