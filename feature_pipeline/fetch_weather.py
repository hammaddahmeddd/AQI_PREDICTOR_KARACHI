"""
fetch_weather.py
----------------
Fetches hourly historical weather data for Karachi from Open-Meteo Archive API.
Saves data directly into MongoDB raw_weather collection.

FIXES APPLIED:
  FIX 1 — Import path: changed `from config import` to a path-safe absolute
           import so this file works whether called from its own directory,
           from the project root, or from run_feature_pipeline.py.

  FIX 2 — Unique index: create_index("datetime", unique=True) is called once
           before the bulk_write so MongoDB enforces no duplicate timestamps
           even if the script is re-run or two jobs overlap.

  FIX 3 — NaN/None safety: pandas float NaN values are replaced with None
           before upsert. PyMongo rejects numpy NaN inside documents with
           a BSON encoding error; None maps cleanly to BSON null.

  FIX 4 — Explicit serverSelectionTimeoutMS=10000 so a bad URI or network
           block fails fast with a clear error instead of hanging for 30s.
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


def fetch_weather() -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        "&hourly="
        "temperature_2m,"
        "relative_humidity_2m,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m,"
        "precipitation,"
        "cloud_cover,"
        "dew_point_2m,"
        "surface_pressure"
        "&wind_speed_unit=kmh"
        "&timezone=Asia%2FKarachi"
    )

    print("\n" + "=" * 80)
    print("WEATHER REQUEST URL")
    print("=" * 80)
    print(url)

    response = requests.get(url, timeout=90)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    data   = response.json()
    hourly = data.get("hourly")

    if hourly is None:
        raise ValueError(f"'hourly' section missing.\nResponse:\n{data}")

    weather_df = pd.DataFrame({
        "datetime":         hourly["time"],
        "temperature":      hourly["temperature_2m"],
        "humidity":         hourly["relative_humidity_2m"],
        "pressure":         hourly["pressure_msl"],
        "wind_speed":       hourly["wind_speed_10m"],
        "wind_direction":   hourly["wind_direction_10m"],
        "wind_gusts":       hourly["wind_gusts_10m"],
        "precipitation":    hourly["precipitation"],
        "cloud_cover":      hourly["cloud_cover"],
        "dew_point":        hourly["dew_point_2m"],
        "surface_pressure": hourly["surface_pressure"],
    })

    # Parse datetime; strip timezone so all dates are naive UTC-equivalent
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    if weather_df["datetime"].dt.tz is not None:
        weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)

    weather_df = (
        weather_df
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"\nRows Retrieved : {len(weather_df):,}")
    print(f"Date Range     : {weather_df['datetime'].min()}  →  {weather_df['datetime'].max()}")

    return weather_df


def save_to_mongodb(df: pd.DataFrame):
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is missing!")

    # FIX 4: explicit timeout — fail fast on bad URI / network block
    client     = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    db         = client["karachi_aqi"]
    collection = db["raw_weather"]

    # FIX 2: enforce uniqueness at the DB layer
    collection.create_index("datetime", unique=True)

    # FIX 3: replace numpy NaN with None so BSON encoding never raises
    records = [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
        for r in df.to_dict(orient="records")
    ]

    if not records:
        client.close()
        return

    print(f"Uploading {len(records):,} weather records to MongoDB (bulk write)...")

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
    print("Weather upload complete!")


if __name__ == "__main__":
    try:
        df = fetch_weather()
        save_to_mongodb(df)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise