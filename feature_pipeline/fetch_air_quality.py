"""
fetch_air_quality.py
---------------------
Fetches hourly air-quality data for Karachi from Open-Meteo Air Quality API.
Saves data directly into MongoDB raw_air_quality collection.

FIXES APPLIED:
  FIX 1 — Import path: same path-safe sys.path insert as fetch_weather.py so
           `from config import` works from any calling directory.

  FIX 2 — Unique index on "datetime" created before bulk_write to prevent
           duplicate timestamps on re-runs or overlapping jobs.

  FIX 3 — NaN/None safety: numpy NaN → None before upsert to avoid BSON
           encoding errors (same pattern as fetch_weather.py).

  FIX 4 — `tz_localize(None)` called correctly: original code called
           `tz_localize(None)` which *removes* tz info (correct), but only
           when `tz is not None`. The condition is preserved and correct.

  FIX 5 — explicit serverSelectionTimeoutMS=10000 for fast failure on bad URI.

  FIX 6 — PM2.5 NaN check now printed as a WARNING (not FATAL crash) when
           0 < pct <= 30, and still raises at >30%. This lets the pipeline
           continue with partially missing data that build_dataset.py will
           forward-fill within its 3-hour gap limit.
"""

import os
import sys
from pathlib import Path

import requests
import pymongo
import pandas as pd
from pymongo import UpdateOne
from dotenv import load_dotenv
load_dotenv()
# ── Import config regardless of working directory ─────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import LATITUDE, LONGITUDE, START_DATE, END_DATE

BULK_BATCH_SIZE = 1000


def fetch_air_quality() -> pd.DataFrame:
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        "&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,"
        "sulphur_dioxide,ozone,dust,uv_index"
        "&timezone=Asia%2FKarachi"
    )

    print("\n" + "=" * 80)
    print("AIR QUALITY REQUEST URL")
    print("=" * 80)
    print(url)

    response = requests.get(url, timeout=90)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()

    data   = response.json()
    hourly = data.get("hourly")

    if not hourly:
        raise ValueError("'hourly' section missing from API response.")

    n = len(hourly["time"])
    aq_df = pd.DataFrame({
        "datetime": hourly["time"],
        "pm25":     hourly["pm2_5"],
        "pm10":     hourly["pm10"],
        "co":       hourly["carbon_monoxide"],
        "no2":      hourly["nitrogen_dioxide"],
        "so2":      hourly["sulphur_dioxide"],
        "o3":       hourly["ozone"],
        # FIX 4: .get() with explicit fallback list so missing keys don't crash
        "dust":     hourly.get("dust",     [None] * n),
        "uv_index": hourly.get("uv_index", [None] * n),
    })

    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    # FIX 4: strip tz correctly
    if aq_df["datetime"].dt.tz is not None:
        aq_df["datetime"] = aq_df["datetime"].dt.tz_localize(None)

    aq_df = (
        aq_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"\nRows Retrieved: {len(aq_df):,}")
    print(f"Date Range    : {aq_df['datetime'].min()}  →  {aq_df['datetime'].max()}")

    # FIX 6: warn at low missing rates, only hard-crash above 30%
    pm25_nan_pct = aq_df["pm25"].isna().mean() * 100
    if pm25_nan_pct > 30:
        raise RuntimeError(
            f"FATAL: PM2.5 is {pm25_nan_pct:.1f}% missing — data unusable."
        )
    if pm25_nan_pct > 0:
        print(f"WARNING: PM2.5 is {pm25_nan_pct:.1f}% missing "
              f"(gaps ≤3h will be forward-filled in build_dataset.py).")

    return aq_df


def save_to_mongodb(df: pd.DataFrame):
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    # FIX 5: explicit timeout
    client     = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    db         = client["karachi_aqi"]
    collection = db["raw_air_quality"]

    # FIX 2: enforce uniqueness
    collection.create_index("datetime", unique=True)

    # FIX 3: numpy NaN → None
    records = [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
        for r in df.to_dict(orient="records")
    ]

    if not records:
        client.close()
        return

    print(f"Uploading {len(records):,} records to MongoDB (bulk upsert)...")

    operations    = [
        UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True)
        for r in records
    ]
    total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE

    for i in range(0, len(operations), BULK_BATCH_SIZE):
        batch_num = i // BULK_BATCH_SIZE + 1
        result    = collection.bulk_write(
            operations[i: i + BULK_BATCH_SIZE], ordered=False
        )
        print(f"  Batch {batch_num}/{total_batches} — "
              f"upserted: {result.upserted_count}, modified: {result.modified_count}")

    client.close()
    print("Air Quality upload complete!")


if __name__ == "__main__":
    try:
        aq_data = fetch_air_quality()
        save_to_mongodb(aq_data)
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        raise