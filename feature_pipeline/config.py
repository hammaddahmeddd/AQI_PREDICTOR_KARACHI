"""
config.py
---------
Karachi coordinates + date range config.

TWO MODES:
  - BACKFILL (run_backfill.py / first-time setup):
      START_DATE = "2023-07-04", END_DATE = yesterday
      Fetches the full 2-year history once.

  - INCREMENTAL (hourly GitHub Actions / feature pipeline):
      START_DATE = 3 days ago, END_DATE = today
      Fetches only the last 3 days to stay fast and avoid rate-limits.
      The 3-day overlap ensures no gaps if the pipeline skipped a run.

ROOT CAUSE FIX:
  The original config always set START_DATE = "2023-07-04" unconditionally,
  causing EVERY hourly pipeline run to re-fetch ~25,000 rows from Open-Meteo.
  This made each run take 5-10 minutes and frequently hit GitHub Actions
  timeout (6 hours), Open-Meteo rate limits, or MongoDB write time limits —
  resulting in the feature store being perpetually stale.

  Now: set PIPELINE_MODE = "incremental" for hourly runs (the default),
  and PIPELINE_MODE = "backfill" only for the one-time historical load.
  The mode can also be set via the PIPELINE_MODE environment variable so
  GitHub Actions can control it without editing this file.
"""

import os
from datetime import datetime, timedelta

LATITUDE  = 24.8607
LONGITUDE = 67.0011

# ── Mode selection ─────────────────────────────────────────────────────────────
# Set PIPELINE_MODE=backfill in env to fetch full history (one-time setup).
# Defaults to "incremental" so hourly runs only fetch the last 3 days.
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "incremental").lower()

today     = datetime.utcnow().date()
yesterday = today - timedelta(days=1)

if PIPELINE_MODE == "backfill":
    # Full historical fetch — run this ONCE manually or via a dedicated workflow job
    START_DATE = "2023-07-04"
    END_DATE   = yesterday.strftime("%Y-%m-%d")
    print(f"[config] BACKFILL mode: {START_DATE} → {END_DATE}")
else:
    # Incremental: only the last 3 days.
    # 3-day window gives a safe overlap buffer if a run was skipped.
    START_DATE = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    END_DATE   = today.strftime("%Y-%m-%d")
    print(f"[config] INCREMENTAL mode: {START_DATE} → {END_DATE}")