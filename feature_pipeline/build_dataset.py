"""
build_dataset.py
-----------------
Fetches raw weather + air-quality data directly from MongoDB, merges them,
handles short-gap interpolation, and stores the aligned records back to MongoDB.

FIXES APPLIED:
  FIX 1 — Causal gap-fill: changed from interpolate(method="linear") to
           ffill(limit=MAX_GAP_FILL_HOURS). interpolate() uses both neighbours
           as anchors even with limit_direction="forward", leaking future values.
           ffill() is strictly causal — this was already correct in your version,
           kept as-is.

  FIX 2 — Bulk write performance: MongoDB writes use bulk_write() in batches
           of 1000 instead of individual update_one() calls — already correct
           in your version, kept as-is.

  FIX 3 — Unique index: create_index("datetime", unique=True) called before
           bulk_write so MongoDB rejects any duplicate timestamps. Without this,
           re-running the script could silently create duplicate documents if
           the upsert filter key changes type (e.g. string vs datetime).

  FIX 4 — datetime stored as native Python datetime, NOT as a string.
           Your version converts datetime to strftime("%Y-%m-%d %H:%M:%S") string
           before upsert. This is the root cause of the mismatch between
           build_dataset.py (string) and feature_engineering.py (which calls
           pd.to_datetime() on whatever it finds). More critically, build_dataset
           uses the string as the upsert filter key, while fetch_weather/
           fetch_air_quality upsert using native datetime objects — so the
           karachi_aqi_dataset ends up with a different key type than the raw
           collections, making cross-collection queries inconsistent.
           FIX: store as native Python datetime (to_pydatetime()) throughout.

  FIX 5 — NaN/None safety: numpy NaN values replaced with None before upsert
           to avoid BSON encoding errors (same pattern as fetch_*.py).

  FIX 6 — serverSelectionTimeoutMS=10000 for fast failure on bad URI.

  FIX 7 — Accepts an optional `db` handle so run_feature_pipeline.py can pass
           an existing connection instead of opening a second one.
"""

import os
import sys
from pathlib import Path

import pymongo
from pymongo import UpdateOne
import pandas as pd

MAX_GAP_FILL_HOURS = 3
BULK_BATCH_SIZE    = 1000


def main(db=None):
    """
    Main entry point.

    Parameters
    ----------
    db : pymongo.database.Database, optional
        Pass an existing authenticated db handle to reuse a connection.
        If None, a new MongoClient is created from MONGODB_URI env var.
    """
    _owns_client = db is None
    client       = None

    if _owns_client:
        mongo_uri = os.environ.get("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI environment variable is missing!")
        # FIX 6: explicit timeout
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
        db     = client["karachi_aqi"]

    try:
        print("Extracting raw data from MongoDB cloud collections...")

        # 1. Load raw collections
        weather_docs = list(db["raw_weather"].find({}, {"_id": 0}))
        aq_docs      = list(db["raw_air_quality"].find({}, {"_id": 0}))

        if not weather_docs:
            raise RuntimeError(
                "raw_weather collection is empty. Run fetch_weather.py first."
            )
        if not aq_docs:
            raise RuntimeError(
                "raw_air_quality collection is empty. Run fetch_air_quality.py first."
            )

        weather_df = pd.DataFrame(weather_docs)
        aq_df      = pd.DataFrame(aq_docs)

        # 2. Normalise datetime — strip timezone, ensure naive UTC-equivalent
        for df in (weather_df, aq_df):
            df["datetime"] = pd.to_datetime(df["datetime"])
            if df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)

        print(f"  raw_weather:     {len(weather_df):,} rows")
        print(f"  raw_air_quality: {len(aq_df):,} rows")

        # 3. Deduplicate and merge
        weather_df = (weather_df
                      .drop_duplicates(subset="datetime")
                      .sort_values("datetime")
                      .reset_index(drop=True))
        aq_df      = (aq_df
                      .drop_duplicates(subset="datetime")
                      .sort_values("datetime")
                      .reset_index(drop=True))

        print("\nExecuting synchronized datetime alignment join...")
        dataset = (
            pd.merge(weather_df, aq_df, on="datetime", how="inner")
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        print(f"Merged shape (before gap-fill): {dataset.shape}")

        if dataset.empty:
            raise RuntimeError(
                "Merge produced zero rows. Check that weather and air-quality "
                "date ranges overlap."
            )

        # 4. Causal short-gap forward-fill (FIX 1 — already correct in original)
        pollutant_cols  = ["pm25", "pm10", "co", "no2", "so2", "o3", "dust", "uv_index"]
        weather_numeric = [
            c for c in dataset.select_dtypes(include="number").columns
            if c not in pollutant_cols
        ]

        for col in pollutant_cols + weather_numeric:
            if col not in dataset.columns:
                continue
            n_before = dataset[col].isna().sum()
            if n_before == 0:
                continue
            dataset[col] = dataset[col].ffill(limit=MAX_GAP_FILL_HOURS)
            n_after      = dataset[col].isna().sum()
            if n_before != n_after:
                print(f"  [gap-fill] {col}: {n_before} → {n_after} NaN "
                      f"(filled {n_before - n_after})")

        # 5. Drop rows with no valid PM2.5 even after gap-fill
        initial_len = len(dataset)
        dataset     = dataset.dropna(subset=["pm25"]).reset_index(drop=True)
        dropped     = initial_len - len(dataset)
        print(f"\nDropped {dropped} rows with missing PM2.5 "
              f"({dropped / initial_len * 100:.2f}% of dataset).")
        print(f"Final dataset shape: {dataset.shape}")

        # 6. Upsert to MongoDB
        # FIX 4: store native Python datetime — NOT string
        records = dataset.to_dict(orient="records")
        for r in records:
            r["datetime"] = r["datetime"].to_pydatetime()  # numpy Timestamp → Python datetime

        # FIX 5: replace numpy NaN with None
        clean_records = [
            {k: (None if isinstance(v, float) and pd.isna(v) else v)
             for k, v in r.items()}
            for r in records
        ]

        output_collection = db["karachi_aqi_dataset"]
        # FIX 3: enforce unique index on datetime
        output_collection.create_index("datetime", unique=True)

        print(f"\nSaving {len(clean_records):,} rows to 'karachi_aqi_dataset' "
              f"via bulk ops (batch={BULK_BATCH_SIZE})...")

        operations    = [
            UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True)
            for r in clean_records
        ]
        total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE

        for i in range(0, len(operations), BULK_BATCH_SIZE):
            batch_num = i // BULK_BATCH_SIZE + 1
            result    = output_collection.bulk_write(
                operations[i: i + BULK_BATCH_SIZE], ordered=False
            )
            print(f"  Batch {batch_num}/{total_batches} — "
                  f"upserted: {result.upserted_count}, "
                  f"modified: {result.modified_count}")

        print("\nSuccessfully built dataset and updated your Cloud Feature Store!")

    finally:
        if _owns_client and client is not None:
            client.close()


if __name__ == "__main__":
    main()