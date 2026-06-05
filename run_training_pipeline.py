"""
run_training_pipeline.py
"""
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pymongo

# ── PATH FIX ──────────────────────────────────────────────────────────────────
# Ensure the root directory is in sys.path so nested modules can find 'load_data'
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Now we can import the pipeline safely
from train_pipeline.train_pipeline import main as run_training

def log(msg: str):
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)

def verify_feature_store(db) -> int:
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

    log("PRE-FLIGHT — Verifying feature store is populated...")
    n_features = verify_feature_store(db)

    if n_features == 0:
        msg = "processed_features collection is empty. Run feature pipeline first."
        log(f"  ✗ ABORT: {msg}")
        push_run_status(db, "ABORTED", msg)
        client.close()
        sys.exit(1)

    log(f"  ✓ Feature store: {n_features:,} documents available.")
    client.close()

    log("Starting consolidated training pipeline...")
    try:
        run_training()
        log("=" * 70)
        log("  TRAINING PIPELINE COMPLETE.")
        log("=" * 70)

        client = pymongo.MongoClient(mongo_uri)
        push_run_status(client["karachi_aqi"], "SUCCESS")
        client.close()
    except Exception as e:
        log(f"  ✗ Training pipeline FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()