"""
feature_engineering.py
-----------------------
Karachi AQI Feature Pipeline — builds the full engineered feature matrix
from raw merged weather + air-quality data and pushes it to MongoDB.

BUGS FIXED IN THIS VERSION
───────────────────────────
  BUG #1 (CRITICAL — data leakage): Step 1 used interpolate(method="time").ffill()
    which is bidirectional. Replaced with ffill() only — strictly causal.

  BUG #2 (CRITICAL — target contamination): targets were computed after
    bidirectional interpolation, so target_aqi_24h etc. were derived from
    future-contaminated pm25. Now computed from cleanly forward-filled data.

  BUG #3 (production safety): _resolve_col() guards wind column name variants.

  BUG #4 (performance): MongoDB writes use bulk_write() in batches of 1000.

  BUG #5 (CRITICAL — XGBoost dtype crash): pd.cut() labels are float literals
    and cast via to_numpy(dtype=np.float64) to guarantee a primitive numeric array.

  BUG #6 (CRITICAL — off-by-one in spike memory): hours_since_aqi_spike and
    consecutive_hours_above_150 were computed from _aqi = df["aqi"].shift(1)
    (already at t-1), then .shift(1) was applied again, producing t-2 values.
    The extra .shift(1) is removed; the loop already operates on lagged AQI.

  BUG #7 (DUPLICATE FEATURES — wastes model capacity, corrupts corr filter):
    The following AQI lag columns were identical to already-existing ones:
      aqi_same_hour_yesterday  == aqi_lag_24   (both shift(24))
      aqi_same_hour_3days_ago  == aqi_lag_72   (both shift(72))
      aqi_same_hour_last_week  == aqi_lag_168  (both shift(168))
      aqi_same_weekday_hour_2w == aqi_same_hour_2weeks_ago == aqi_lag_336 (shift(336))
    Exact duplicates removed; named aliases retained only for unique shift values
    (720h, 672h) that were not already in the numeric lag grid.

NEW FEATURES ADDED FOR R² IMPROVEMENT
───────────────────────────────────────
  FEAT #1: human_emissions_proxy — explicit float weight for weekday/rush-hour.

  FEAT #2: Atmospheric stagnation features — diurnal_temp_range_24h,
    temp_to_wind_ratio, humidity_to_wind_ratio, is_atmospheric_stagnant.

  FEAT #3: Target deviation targets (alternative training targets only).

  FEAT #4: Extended AQI EWM spans (48h, 72h, 168h) for longer-horizon models.
    EWM(24) decays too fast to carry signal into 48h/72h predictions.

  FEAT #5: pm25_roll_q90_72h — rolling 90th-percentile of PM2.5 over 72h.
    Captures the upper tail of recent pollution episodes; strong spike precursor.

  FEAT #6: heat_index_proxy = temperature * humidity / 100. Hygroscopic growth
    of PM2.5 increases with T×RH; this interaction is non-linear and must be
    made explicit for tree models.

  FEAT #7: Karachi sea-breeze proxy — wind_from_sea flag (direction 180°–270°,
    Arabian Sea quadrant). Sea-breeze onset suppresses PM2.5 in coastal Karachi
    and is a dominant local meteorological driver unavailable from scalars alone.

  FEAT #8: pm25_ewm_72, pm25_ewm_168 for longer-range PM2.5 trend signal.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymongo
from pymongo import UpdateOne

# Path setup: insert both this file's directory AND the project root so
# imports work from any working directory.
PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR     = PIPELINE_DIR.parent
for _p in (str(PIPELINE_DIR), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BULK_BATCH_SIZE = 1000


# ── 1. EPA AQI Calculation Helpers ────────────────────────────────────────────

def calculate_aqi_from_pm25(pm25: float) -> float:
    """US-EPA piecewise linear interpolation: PM2.5 (µg/m³) → AQI."""
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    breakpoints = [
        (0.0,   12.0,  0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            )
    return 500


def aqi_to_category(aqi: float) -> float:
    if pd.isna(aqi): return np.nan
    if aqi <= 50:    return 0.0
    if aqi <= 100:   return 1.0
    if aqi <= 150:   return 2.0
    if aqi <= 200:   return 3.0
    if aqi <= 300:   return 4.0
    return 5.0


def _rolling_slope_6h(y: np.ndarray) -> float:
    """OLS slope of a 6-point window — encodes short-term AQI trend direction."""
    x_dev = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    return float(np.dot(x_dev, y - np.mean(y)) / 17.5)


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Returns the first candidate column name present in df, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# ── 2. Core Feature Builder ───────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "datetime" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Building features. Input shape: {df.shape}")

    # ── STEP 1: CAUSAL FORWARD-FILL IMPUTATION ───────────────────────────────
    # ffill() only — no bfill, no interpolate. Strictly causal.
    # Leading NaNs at the series start are handled by the warm-up filter (Step 15).
    df = df.set_index("datetime")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].ffill()
    df = df.reset_index()
    print(" -> Causal ffill imputation complete.")

    # ── STEP 2: AQI COMPUTATION + MULTI-HORIZON TARGETS ─────────────────────
    df["aqi"] = df["pm25"].apply(calculate_aqi_from_pm25)

    new_cols = {}
    for h in [12, 24, 48, 72]:
        new_cols[f"target_aqi_{h}h"]     = df["aqi"].shift(-h)
        new_cols[f"target_aqi_{h}h_log"] = np.log1p(new_cols[f"target_aqi_{h}h"])
        new_cols[f"target_cat_{h}h"]     = new_cols[f"target_aqi_{h}h"].apply(aqi_to_category)

    # FEAT #3: deviation targets — mean-reverting alternative targets (never features)
    # 7-day (168h) rolling median of lag-1 AQI gives a causal structural baseline.
    # Listed in ALL_TARGETS in load_data.py; explicitly excluded from X before training.
    _aqi_historical_anchor = (
        df["aqi"].shift(1)
        .rolling(168, min_periods=24)
        .median()
        .fillna(df["aqi"].median())
    )
    for h in [12, 24, 48, 72]:
        new_cols[f"target_aqi_{h}h_deviation"] = df["aqi"].shift(-h) - _aqi_historical_anchor

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── STEP 3: TEMPORAL EMBEDDINGS ──────────────────────────────────────────
    dt        = df["datetime"]
    hour      = dt.dt.hour
    month     = dt.dt.month
    weekday   = dt.dt.weekday
    day_of_yr = dt.dt.dayofyear

    temp_cols = {
        "hour":         hour,
        "day":          dt.dt.day,
        "month":        month,
        "weekday":      weekday,
        "week_of_year": dt.dt.isocalendar().week.astype(int).values,
        "quarter":      dt.dt.quarter,
        "day_of_year":  day_of_yr,
        # Cyclic embeddings — prevents discontinuity at boundary (e.g. hour 23→0)
        "hour_sin":     np.sin(2 * np.pi * hour    / 24),
        "hour_cos":     np.cos(2 * np.pi * hour    / 24),
        "month_sin":    np.sin(2 * np.pi * month   / 12),
        "month_cos":    np.cos(2 * np.pi * month   / 12),
        "weekday_sin":  np.sin(2 * np.pi * weekday / 7),
        "weekday_cos":  np.cos(2 * np.pi * weekday / 7),
        "doy_sin":      np.sin(2 * np.pi * day_of_yr / 365),
        "doy_cos":      np.cos(2 * np.pi * day_of_yr / 365),
        "is_weekend":   (weekday >= 5).astype(int),
        "is_rush_hour": hour.isin([7, 8, 9, 17, 18, 19]).astype(int),
        "hour_of_week": weekday * 24 + hour,
        # FEAT #1: continuous emission weight — more granular than a binary flag.
        # 1.0 = weekday rush, 0.7 = weekday off-peak, 0.3 = weekend/holiday.
        "human_emissions_proxy": np.where(
            (weekday < 5) & hour.isin([8, 9, 17, 18, 19]), 1.0,
            np.where(weekday < 5, 0.7, 0.3)
        ),
    }
    df = pd.concat([df, pd.DataFrame(temp_cols, index=df.index)], axis=1)

    # ── STEP 4: WIND VECTOR DECOMPOSITION ───────────────────────────────────
    ws_col = _resolve_col(df, "wind_speed", "wind_speed_10m")
    wd_col = _resolve_col(df, "wind_direction", "wind_direction_10m")

    if ws_col and wd_col:
        wdir_rad = np.deg2rad(df[wd_col].shift(1))
        ws_lag   = df[ws_col].shift(1)
        wind_cols = {
            "wind_dir_sin": np.sin(wdir_rad),
            "wind_dir_cos": np.cos(wdir_rad),
            "wind_x":       ws_lag * np.sin(wdir_rad),
            "wind_y":       ws_lag * np.cos(wdir_rad),
        }
        df = pd.concat([df, pd.DataFrame(wind_cols, index=df.index)], axis=1)

        # FEAT #7: Karachi sea-breeze proxy
        # Arabian Sea lies SW–W of Karachi; onshore flow (180°–270°) brings clean
        # maritime air and suppresses PM2.5. This quadrant flag is a strong local
        # predictor that cannot be derived from wind speed alone.
        _wd_lag = df[wd_col].shift(1)
        df["wind_from_sea"] = (
            (_wd_lag >= 180) & (_wd_lag <= 270)
        ).astype(float)
        # Interaction: sea breeze × wind speed → dispersion power from the sea
        df["sea_breeze_strength"] = df["wind_from_sea"] * ws_lag.fillna(0)
        print(f" -> Wind decomposition + sea-breeze proxy: '{ws_col}' + '{wd_col}'.")
    else:
        print(" -> WARNING: Wind columns not found — Steps 4/7 skipped.")

    # ── STEP 5: AQI LAG CHAINS ───────────────────────────────────────────────
    # BUG #7 FIX: removed duplicates.
    # shift(24)=aqi_lag_24, shift(72)=aqi_lag_72, shift(168)=aqi_lag_168,
    # shift(336)=aqi_lag_336 already exist in the numeric grid below.
    # Only genuinely unique named lags are kept: shift(720) and shift(672).
    _aqi = df["aqi"].shift(1)
    lag_cols = {f"aqi_lag_{lag}": df["aqi"].shift(lag)
                for lag in [1, 2, 3, 6, 7, 12, 24, 48, 72, 168, 336]}
    lag_cols.update({
        # shift(720) = 30 days — not in the numeric grid above
        "aqi_same_hour_30days_ago":  df["aqi"].shift(720),
        # shift(672) = 28 days / 4 exact weekday cycles — not in the numeric grid
        "aqi_same_weekday_hour_4w":  df["aqi"].shift(672),
    })
    df = pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)

    # ── STEP 6: PM2.5 ROLLING FEATURES ──────────────────────────────────────
    _pm25 = df["pm25"].shift(1)
    pm25_lags = {f"pm25_lag_{lag}": df["pm25"].shift(lag)
                 for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]}
    pm25_rolls = {}
    for w in [6, 24, 72, 168, 336]:
        pm25_rolls[f"pm25_roll_mean_{w}"] = _pm25.rolling(w, min_periods=1).mean()
        pm25_rolls[f"pm25_roll_std_{w}"]  = _pm25.rolling(w, min_periods=1).std().fillna(0)
    pm25_rolls.update({
        "pm25_roll_max_24":  _pm25.rolling(24,  min_periods=1).max(),
        "pm25_roll_max_72":  _pm25.rolling(72,  min_periods=1).max(),
        # FEAT #5: rolling 90th percentile — captures upper tail without
        # the noise of rolling max (which reacts to single outlier hours)
        "pm25_roll_q90_72h": _pm25.rolling(72,  min_periods=12).quantile(0.90).fillna(_pm25),
        "pm25_ewm_24":       _pm25.ewm(span=24,  adjust=False).mean(),
        # FEAT #8: longer EWM spans for 48h/72h horizon models
        "pm25_ewm_72":       _pm25.ewm(span=72,  adjust=False).mean(),
        "pm25_ewm_168":      _pm25.ewm(span=168, adjust=False).mean(),
        "pm25_change_24h":   df["pm25"].shift(1) - df["pm25"].shift(25),
        "pm25_vs_24h_avg":   _pm25 - _pm25.rolling(24, min_periods=1).mean(),
        "pm25_vs_72h_avg":   _pm25 - _pm25.rolling(72, min_periods=1).mean(),
    })
    df = pd.concat([df, pd.DataFrame({**pm25_lags, **pm25_rolls}, index=df.index)], axis=1)

    # ── STEP 7: PM10 ROLLING FEATURES ───────────────────────────────────────
    _pm10 = df["pm10"].shift(1)
    pm10_cols = {f"pm10_lag_{lag}": df["pm10"].shift(lag) for lag in [1, 6, 24, 72]}
    pm10_cols.update({
        "pm10_roll_mean_24": _pm10.rolling(24, min_periods=1).mean(),
        "pm10_roll_mean_72": _pm10.rolling(72, min_periods=1).mean(),
        "pm10_roll_std_24":  _pm10.rolling(24, min_periods=1).std().fillna(0),
        "pm10_change_24h":   df["pm10"].shift(1) - df["pm10"].shift(25),
        "dust_event":        (_pm10 > 250).astype(int),
    })
    df = pd.concat([df, pd.DataFrame(pm10_cols, index=df.index)], axis=1)

    # ── STEP 8: AQI ROLLING MATRIX ───────────────────────────────────────────
    aqi_roll = {}
    for w in [6, 12, 24, 48, 72, 168, 336]:
        aqi_roll[f"aqi_roll_mean_{w}"] = _aqi.rolling(w, min_periods=1).mean()
    for w in [24, 72, 168]:
        aqi_roll[f"aqi_roll_std_{w}"] = _aqi.rolling(w, min_periods=1).std().fillna(0)
        aqi_roll[f"aqi_roll_max_{w}"] = _aqi.rolling(w, min_periods=1).max()
        aqi_roll[f"aqi_roll_min_{w}"] = _aqi.rolling(w, min_periods=1).min()
    aqi_roll.update({
        "aqi_ewm_24":  _aqi.ewm(span=24,  adjust=False).mean(),
        "aqi_ewm_72":  _aqi.ewm(span=72,  adjust=False).mean(),
        # FEAT #4: longer EWM spans — carry trend signal further into the future
        "aqi_ewm_168": _aqi.ewm(span=168, adjust=False).mean(),
    })
    df = pd.concat([df, pd.DataFrame(aqi_roll, index=df.index)], axis=1)

    # ── STEP 9: MOMENTUM & TREND ─────────────────────────────────────────────
    r24  = df["aqi_roll_mean_24"]
    r72  = df["aqi_roll_mean_72"]
    r168 = df["aqi_roll_mean_168"]
    p24  = df["pm25_roll_mean_24"]
    p72  = df["pm25_roll_mean_72"]
    mom_cols = {
        "aqi_trend_ratio":     r24 / (r72 + 1),
        "pm25_trend_ratio":    p24 / (p72 + 1),
        "aqi_momentum_6_24":   df["aqi_roll_mean_6"] - r24,
        "aqi_momentum_24_72":  r24 - r72,
        "aqi_momentum_24_168": r24 - r168,
        "aqi_change_1h":       _aqi - df["aqi"].shift(2),
        # shift(1) - shift(7) = AQI 6h ago relative to 1h ago = 6h change ✓
        "aqi_change_6h":       _aqi - df["aqi"].shift(7),
        "aqi_change_24h":      _aqi - df["aqi"].shift(25),
        "aqi_trend_slope_6h":  (
            _aqi.rolling(6, min_periods=6)
                .apply(_rolling_slope_6h, raw=True)
                .fillna(0)
        ),
        "aqi_acceleration":    _aqi.diff().diff().fillna(0),
    }
    df = pd.concat([df, pd.DataFrame(mom_cols, index=df.index)], axis=1)

    # ── STEP 10: PERSISTENCE & RECOVERY ──────────────────────────────────────
    persist_cols = {
        "aqi_persistence":       df["aqi_lag_1"] - df["aqi_lag_24"],
        "aqi_diff_24_168":       df["aqi_lag_24"] - df["aqi_lag_168"],
        "aqi_recovery_rate":     (df["aqi_roll_max_72"] - df["aqi_lag_1"]) / (df["aqi_roll_max_72"] + 1),
        "aqi_persistence_ratio": df["aqi_lag_1"] / (df["aqi_roll_mean_72"] + 1),
        "aqi_vs_week":           df["aqi_roll_mean_24"] - df["aqi_roll_mean_168"],
    }
    df = pd.concat([df, pd.DataFrame(persist_cols, index=df.index)], axis=1)

    # ── STEP 11: STATISTICAL ANOMALY DETECTION ───────────────────────────────
    _rm72 = _aqi.rolling(72,  min_periods=24).mean()
    _rs72 = _aqi.rolling(72,  min_periods=24).std().replace(0, 1)
    _rq90 = _aqi.rolling(168, min_periods=72).quantile(0.90)

    # BUG #5 FIX: float labels + explicit np.float64 cast avoids Categorical dtype
    _aqi_regime_cat = pd.cut(
        df["aqi_lag_1"],
        bins=[0, 50, 100, 150, 200, 300, 1000],
        labels=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    _aqi_regime = pd.Series(
        _aqi_regime_cat.to_numpy(dtype=np.float64, na_value=np.nan),
        index=df.index,
    ).fillna(0.0)

    anomaly_cols = {
        "aqi_zscore_72h":       ((df["aqi_lag_1"] - _rm72) / _rs72).fillna(0),
        "aqi_percentile_72":    (
            _aqi.rolling(72, min_periods=12)
                .apply(lambda x: float(np.mean(x < x[-1])), raw=True)
                .fillna(0.5)
        ),
        "aqi_above_recent_q90": (df["aqi_lag_1"] > _rq90).astype(int),
        "aqi_volatility_ratio": df["aqi_roll_std_24"] / (df["aqi_roll_std_72"] + 1),
        "aqi_regime":           _aqi_regime,
    }
    df = pd.concat([df, pd.DataFrame(anomaly_cols, index=df.index)], axis=1)

    # ── STEP 12: SPIKE MEMORY ────────────────────────────────────────────────
    # BUG #6 FIX: _aqi = df["aqi"].shift(1) is already at t-1.
    # The old code applied .shift(1) again to the computed series, pushing to t-2.
    # Removed the extra .shift(1) — the loop result is already causally correct.
    spike_flag = (_aqi > 150)
    _hours_since, _consec = [], []
    _counter, _run = 999, 0
    for s in spike_flag:
        if s:
            _counter = 0
            _run    += 1
        else:
            _counter += 1
            _run      = 0
        _hours_since.append(_counter)
        _consec.append(_run)

    spike_cols = {
        "spike_count_72h":             spike_flag.rolling(72, min_periods=1).sum().fillna(0),
        "dust_hours_72h":              (_pm10 > 250).rolling(72, min_periods=1).sum().fillna(0),
        # BUG #6 FIX: no extra .shift(1) here — _aqi is already shift(1)
        "hours_since_aqi_spike":       pd.Series(_hours_since, index=df.index),
        "consecutive_hours_above_150": pd.Series(_consec,      index=df.index),
    }
    df = pd.concat([df, pd.DataFrame(spike_cols, index=df.index)], axis=1)

    # ── STEP 13: SECONDARY POLLUTANT PROXIES ─────────────────────────────────
    poll_cols = {}
    for poll in ["so2", "co", "o3", "no2"]:
        if poll not in df.columns:
            continue
        _s = df[poll].shift(1)
        poll_cols[f"{poll}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        poll_cols[f"{poll}_change_24h"]   = df[poll].shift(1) - df[poll].shift(25)

    for poll in ["no2", "o3"]:
        if poll not in df.columns:
            continue
        poll_cols[f"{poll}_roll_std_24"] = (
            df[poll].shift(1).rolling(24, min_periods=1).std().fillna(0)
        )

    if {"pm25", "pm10"}.issubset(df.columns):
        poll_cols["pm25_pm10_ratio"] = (
            df["pm25"].shift(1) / (df["pm10"].shift(1) + 1e-3)
        ).clip(upper=10)
        poll_cols["pm25_fraction"] = (
            df["pm25"].shift(1)
            / (df["pm25"].shift(1) + df["pm10"].shift(1) + 1e-3)
        )

    if {"no2", "o3"}.issubset(df.columns):
        poll_cols["no2_o3_ratio"] = df["no2"].shift(1) / (df["o3"].shift(1) + 1e-3)

    if "dust" in df.columns:
        _dust = df["dust"].shift(1)
        poll_cols["dust_lag_1"]        = _dust
        poll_cols["dust_roll_mean_24"] = _dust.rolling(24, min_periods=1).mean()
        poll_cols["dust_roll_max_24"]  = _dust.rolling(24, min_periods=1).max()

    if "uv_index" in df.columns:
        poll_cols["uv_lag_1"]        = df["uv_index"].shift(1)
        poll_cols["uv_roll_mean_24"] = df["uv_index"].shift(1).rolling(24, min_periods=1).mean()

    df = pd.concat([df, pd.DataFrame(poll_cols, index=df.index)], axis=1)

    # ── STEP 14: METEOROLOGICAL DISPERSAL & INTERACTIONS ─────────────────────
    met_cols  = {}
    temp_col  = _resolve_col(df, "temperature",  "temperature_2m")
    hum_col   = _resolve_col(df, "humidity",      "relative_humidity_2m")
    ws_col_14 = _resolve_col(df, "wind_speed",    "wind_speed_10m")

    # Per-variable lag + rolling stats
    for col, resolved in [
        ("temperature", temp_col),
        ("humidity",    hum_col),
        ("wind_speed",  ws_col_14),
    ]:
        if not resolved:
            print(f" -> WARNING: '{col}' not found — met rolling features skipped.")
            continue
        _s = df[resolved].shift(1)
        met_cols[f"{col}_roll_mean_24"] = _s.rolling(24, min_periods=1).mean()
        met_cols[f"{col}_change_24h"]   = df[resolved].shift(1) - df[resolved].shift(25)
        for lag in [1, 6, 24]:
            met_cols[f"{col}_lag_{lag}"] = df[resolved].shift(lag)

    if ws_col_14:
        ws_lag1    = df[ws_col_14].shift(1)
        ws_roll24  = ws_lag1.rolling(24,  min_periods=1).mean()
        ws_roll168 = ws_lag1.rolling(168, min_periods=1).mean()
        met_cols["wind_speed_roll_std_24"]  = ws_lag1.rolling(24, min_periods=1).std().fillna(0)
        met_cols["wind_persistence_ratio"]  = ws_roll24 / (ws_roll168 + 0.1)

    if temp_col:
        met_cols["temperature_roll_std_24"] = (
            df[temp_col].shift(1).rolling(24, min_periods=1).std().fillna(0)
        )

    if temp_col and hum_col:
        t_l1 = df[temp_col].shift(1)
        h_l1 = df[hum_col].shift(1)
        met_cols["temp_humidity"]    = t_l1 * h_l1
        met_cols["heat_dryness"]     = t_l1 / (h_l1 + 1)
        # FEAT #6: heat_index_proxy — T×RH/100 drives hygroscopic PM2.5 growth.
        # Dividing by 100 keeps it on a 0–100 scale comparable to temperature.
        met_cols["heat_index_proxy"] = t_l1 * h_l1 / 100.0

    if ws_col_14 and "pm25" in df.columns:
        met_cols["wind_dispersal"]               = df[ws_col_14].shift(1) / (df["pm25"].shift(1) + 5)
        # interaction_pm25_wind_inverse: high ratio = low wind + high PM2.5 = accumulation regime
        met_cols["interaction_pm25_wind_inverse"] = (
            df["pm25"].shift(1) / (df[ws_col_14].shift(1) + 0.5)
        ).clip(upper=500)

    if hum_col and "pm25" in df.columns:
        # interaction_pm25_humidity: humid air hygroscopically grows particles
        met_cols["interaction_pm25_humidity"] = (
            df["pm25"].shift(1) * df[hum_col].shift(1) / 100.0
        )

    dew_col = _resolve_col(df, "dew_point", "dew_point_2m")
    if temp_col and dew_col:
        met_cols["dew_point_depression"]     = df[temp_col].shift(1) - df[dew_col].shift(1)
        met_cols["dew_pt_depression_roll24"] = (
            met_cols["dew_point_depression"].rolling(24, min_periods=1).mean()
        )

    # FEAT #2: Atmospheric stagnation features
    # Physical basis: when wind < 2 m/s AND air is nearly saturated (small dew-point
    # depression), the planetary boundary layer collapses and PM2.5 cannot disperse.
    # Making these thresholds explicit halves the split depth tree models need.
    if temp_col and hum_col and ws_col_14:
        t_lag1       = df[temp_col].shift(1)
        ws_lag1_stag = df[ws_col_14].shift(1)

        met_cols["diurnal_temp_range_24h"] = (
            t_lag1.rolling(24, min_periods=6).max()
            - t_lag1.rolling(24, min_periods=6).min()
        ).fillna(0.0)

        met_cols["temp_to_wind_ratio"]     = t_lag1 / (ws_lag1_stag + 0.1)
        met_cols["humidity_to_wind_ratio"] = df[hum_col].shift(1) / (ws_lag1_stag + 0.1)

        if "dew_point_depression" in met_cols:
            met_cols["is_atmospheric_stagnant"] = (
                (ws_lag1_stag < 2.0) & (met_cols["dew_point_depression"] < 3.0)
            ).astype(float)

    if "precipitation" in df.columns:
        prec = df["precipitation"].shift(1)
        met_cols["rain_24h"]   = prec.rolling(24, min_periods=1).sum()
        met_cols["rain_72h"]   = prec.rolling(72, min_periods=1).sum()
        met_cols["rain_event"] = (prec > 0.1).astype(int)

    if "pressure" in df.columns:
        pres = df["pressure"].shift(1)
        met_cols["pressure_lag_1"]        = pres
        met_cols["pressure_roll_mean_24"] = pres.rolling(24, min_periods=1).mean()
        met_cols["pressure_change_24h"]   = pres - df["pressure"].shift(25)
        met_cols["pressure_change_6h"]    = pres - df["pressure"].shift(7)

    if "surface_pressure" in df.columns:
        met_cols["surface_pressure_lag_1"]      = df["surface_pressure"].shift(1)
        met_cols["surface_pressure_change_24h"] = (
            df["surface_pressure"].shift(1) - df["surface_pressure"].shift(25)
        )

    if "wind_gusts" in df.columns:
        _gust = df["wind_gusts"].shift(1)
        met_cols["wind_gusts_roll_mean_24"] = _gust.rolling(24, min_periods=1).mean()
        met_cols["wind_gusts_roll_max_24"]  = _gust.rolling(24, min_periods=1).max()

    if "cloud_cover" in df.columns:
        met_cols["cloud_cover_lag_1"]        = df["cloud_cover"].shift(1)
        met_cols["cloud_cover_roll_mean_24"] = (
            df["cloud_cover"].shift(1).rolling(24, min_periods=1).mean()
        )

    df = pd.concat([df, pd.DataFrame(met_cols, index=df.index)], axis=1)

    # ── STEP 15: WARM-UP ROW FILTERING ──────────────────────────────────────
    # aqi_same_hour_30days_ago requires 720 warm-up rows (30 days).
    # target_aqi_72h requires the final 72 rows to be dropped (no future data).
    # Both are non-negotiable — rows failing these are unusable for training.
    required_non_null = ["aqi_same_hour_30days_ago", "target_aqi_72h"]
    before = len(df)
    df = df.dropna(subset=required_non_null).reset_index(drop=True)
    print(f" -> Dropped {before - len(df):,} warm-up rows. {len(df):,} rows remaining.")

    assert df["datetime"].is_monotonic_increasing, "CRITICAL: Temporal order broken."

    # ── STEP 16: DTYPE SAFETY PASS ───────────────────────────────────────────
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    non_date_obj = [c for c in obj_cols if c != "datetime"]
    if non_date_obj:
        print(f" -> WARNING: Coercing {len(non_date_obj)} non-numeric columns to float64: {non_date_obj}")
        for col in non_date_obj:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64).fillna(0.0)

    # Final feature count report
    feature_cols = [c for c in df.columns if c not in
                    ["datetime", "timestamp"] and not c.startswith("target_")]
    print(f" -> Feature matrix: {len(feature_cols)} engineered features, {len(df):,} rows.")
    return df


# ── 3. Production Pipeline Entrypoint ─────────────────────────────────────────

def process_all(db=None):
    """
    Build the engineered feature matrix from karachi_aqi_dataset and upsert
    results into processed_features.

    Parameters
    ----------
    db : pymongo.database.Database, optional
        Pass an existing authenticated db handle to reuse a connection.
        If None, a new MongoClient is created from MONGODB_URI env var.

    FIXES vs original:
      FIX A - datetime stored as native Python datetime, NOT as string.
              Original .strftime() converted datetimes to strings causing:
              (a) upsert filter key type mismatch vs raw collections,
              (b) duplicate documents on re-runs (filter never matched existing docs),
              (c) load_data.py had to re-parse strings back to datetime every time.
      FIX B - NaN/None safety: numpy NaN replaced with None before upsert.
      FIX C - Unique index on "datetime" before bulk_write prevents duplicates.
      FIX D - serverSelectionTimeoutMS=10000 for fast failure on bad URI.
      FIX E - Accepts optional db handle so run_feature_pipeline.py can reuse
              its existing connection instead of opening a second one.
    """
    _owns_client = db is None
    client       = None

    if _owns_client:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("CRITICAL: MONGODB_URI missing from environment.")
        # FIX D
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
        db     = client["karachi_aqi"]

    try:
        print("\n" + "=" * 70)
        print(" EXTRACTING RAW DATASET FROM MONGODB")
        print("=" * 70)

        cursor = db["karachi_aqi_dataset"].find({}, {"_id": 0})
        raw_df = pd.DataFrame(list(cursor))

        if raw_df.empty:
            raise RuntimeError(
                "CRITICAL: 'karachi_aqi_dataset' is empty. Run build_dataset.py first."
            )

        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
        if raw_df["datetime"].dt.tz is not None:
            raw_df["datetime"] = raw_df["datetime"].dt.tz_localize(None)

        processed_df = build_features(raw_df)

        print("\n" + "=" * 70)
        print(" WRITING FEATURE STORE TO MONGODB")
        print("=" * 70)

        # FIX A: keep datetime as native Python datetime - do NOT strftime() to string.
        # numpy Timestamp.to_pydatetime() gives a proper Python datetime for BSON.
        mongo_df = processed_df.copy()
        mongo_df["timestamp"] = mongo_df["datetime"].apply(lambda x: x.to_pydatetime())
        mongo_df["datetime"]  = mongo_df["datetime"].apply(lambda x: x.to_pydatetime())

        features_payload = mongo_df.to_dict(orient="records")

        # FIX B: replace numpy NaN with None
        clean_payload = [
            {k: (None if isinstance(v, float) and pd.isna(v) else v)
             for k, v in doc.items()}
            for doc in features_payload
        ]

        print(f"Prepared {len(clean_payload):,} documents.")

        output_collection = db["processed_features"]
        # FIX C: enforce uniqueness before writing
        output_collection.create_index("datetime", unique=True)

        operations = [
            UpdateOne(
                {"datetime": r["datetime"]},
                {"$set": r},
                upsert=True,
            )
            for r in clean_payload
        ]

        total_batches = (len(operations) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE
        for i in range(0, len(operations), BULK_BATCH_SIZE):
            batch_num = i // BULK_BATCH_SIZE + 1
            result = output_collection.bulk_write(
                operations[i: i + BULK_BATCH_SIZE],
                ordered=False,
            )
            print(f"  Batch {batch_num}/{total_batches} \u2014 "
                  f"upserted: {result.upserted_count}, modified: {result.modified_count}")

        print("\nFeature store synchronised successfully.")
        print("=" * 70 + "\n")

    finally:
        if _owns_client and client is not None:
            client.close()

if __name__ == "__main__":
    process_all()