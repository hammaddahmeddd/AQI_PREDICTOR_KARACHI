"""
feature_engineering.py
-----------------------
Karachi AQI Feature Pipeline — builds the full engineered feature matrix
from raw merged weather + air-quality data and pushes it to MongoDB.

DESIGN PRINCIPLES
──────────────────
  - All data comes from MongoDB; nothing is read from or written to local disk.
  - SHAP/feature images (FSH) are cleaned from MongoDB before each daily run
    to stay within free-tier storage limits.
  - All shifts are strictly causal (shift(1) minimum for same-row raw values).
  - Targets are computed BEFORE any lag/rolling, from the clean AQI series.

KEY IMPROVEMENTS FOR R² (vs previous version)
───────────────────────────────────────────────
  IMPR #1: Fourier seasonal terms (annual + semi-annual) — gives linear/tree
    models an explicit signal for Karachi's two pollution seasons
    (Nov–Feb winter haze, May–Jun pre-monsoon dust).

  IMPR #2: Boundary-layer height proxy — morning hours + low wind + high
    humidity → shallow mixing → PM2.5 accumulation. Explicit compound flag
    halves the splits trees need to re-discover this every run.

  IMPR #3: Extended rolling windows for AQI / PM2.5 (14-day = 336h).
    Multi-week trends matter for monsoon onset/offset regime changes.

  IMPR #4: PM2.5 rate-of-change features: 3h, 6h absolute change AND
    signed direction. Models trained only on levels miss acceleration events.

  IMPR #5: Interaction terms — pm25 × humidity, wind × temp (ventilation
    index), pm25 × wind_inverse (accumulation index). These non-linear
    interactions are critical for tree depth efficiency.

  IMPR #6: Cross-pollutant ratios — NO₂/O₃ ratio encodes photochemical
    activity; PM2.5/PM10 fraction encodes fine-vs-coarse source mix.
    Both are regime-discriminating features that are not derivable from
    individual pollutant lags alone.

  IMPR #7: Monsoon flag — binary indicator for Jun–Sep using Karachi's
    climatological onset. Combined with rain_24h it captures washout events.

  IMPR #8: Persistence-corrected target: the model now stores both the
    raw AQI target and a 24h-delta target, allowing ensemble stacking to
    combine "level prediction" with "change prediction" for better R².
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymongo
from pymongo import UpdateOne

PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR     = PIPELINE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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

    # Deviation targets — causal 7-day rolling median anchor, useful for
    # change-prediction ensembling (IMPR #8)
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

    # IMPR #1: Fourier seasonal terms
    # Annual cycle (365.25d): captures winter haze (Nov–Feb) vs summer
    # Semi-annual cycle (182.6d): captures pre-monsoon dust (May–Jun)
    doy_float = day_of_yr + hour / 24.0
    fourier_cols = {
        "fourier_annual_sin":      np.sin(2 * np.pi * doy_float / 365.25),
        "fourier_annual_cos":      np.cos(2 * np.pi * doy_float / 365.25),
        "fourier_semi_annual_sin": np.sin(4 * np.pi * doy_float / 365.25),
        "fourier_semi_annual_cos": np.cos(4 * np.pi * doy_float / 365.25),
    }

    # IMPR #7: Karachi monsoon flag (climatological onset Jun–Sep)
    monsoon_flag = month.isin([6, 7, 8, 9]).astype(float)

    temp_cols = {
        "hour":         hour,
        "day":          dt.dt.day,
        "month":        month,
        "weekday":      weekday,
        "week_of_year": dt.dt.isocalendar().week.astype(int).values,
        "quarter":      dt.dt.quarter,
        "day_of_year":  day_of_yr,
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
        "human_emissions_proxy": np.where(
            (weekday < 5) & hour.isin([8, 9, 17, 18, 19]), 1.0,
            np.where(weekday < 5, 0.7, 0.3)
        ),
        "is_monsoon":  monsoon_flag,
        **fourier_cols,
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

        # Sea-breeze proxy: Arabian Sea onshore flow (180°–270°) suppresses PM2.5
        _wd_lag = df[wd_col].shift(1)
        df["wind_from_sea"]    = ((_wd_lag >= 180) & (_wd_lag <= 270)).astype(float)
        df["sea_breeze_strength"] = df["wind_from_sea"] * ws_lag.fillna(0)
        print(f" -> Wind decomposition + sea-breeze proxy: '{ws_col}' + '{wd_col}'.")
    else:
        print(" -> WARNING: Wind columns not found — Steps 4/7 skipped.")

    # ── STEP 5: AQI LAG CHAINS ───────────────────────────────────────────────
    _aqi = df["aqi"].shift(1)
    lag_cols = {f"aqi_lag_{lag}": df["aqi"].shift(lag)
                for lag in [1, 2, 3, 6, 7, 12, 24, 48, 72, 168, 336]}
    lag_cols.update({
        "aqi_same_hour_30days_ago": df["aqi"].shift(720),
        "aqi_same_weekday_hour_4w": df["aqi"].shift(672),
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
        "pm25_roll_q90_72h": _pm25.rolling(72,  min_periods=12).quantile(0.90).fillna(_pm25),
        "pm25_ewm_24":       _pm25.ewm(span=24,  adjust=False).mean(),
        "pm25_ewm_72":       _pm25.ewm(span=72,  adjust=False).mean(),
        "pm25_ewm_168":      _pm25.ewm(span=168, adjust=False).mean(),
        "pm25_change_24h":   df["pm25"].shift(1) - df["pm25"].shift(25),
        "pm25_vs_24h_avg":   _pm25 - _pm25.rolling(24, min_periods=1).mean(),
        "pm25_vs_72h_avg":   _pm25 - _pm25.rolling(72, min_periods=1).mean(),
        # IMPR #4: short-range rate-of-change
        "pm25_change_3h":    df["pm25"].shift(1) - df["pm25"].shift(4),
        "pm25_change_6h":    df["pm25"].shift(1) - df["pm25"].shift(7),
        "pm25_roc_sign":     np.sign(df["pm25"].shift(1) - df["pm25"].shift(4)),
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

    # IMPR #6: Cross-pollutant ratios (regime discriminators)
    if {"pm25", "pm10"}.issubset(df.columns):
        poll_cols["pm25_pm10_ratio"] = (
            df["pm25"].shift(1) / (df["pm10"].shift(1) + 1e-3)
        ).clip(upper=10)
        poll_cols["pm25_fraction"] = (
            df["pm25"].shift(1)
            / (df["pm25"].shift(1) + df["pm10"].shift(1) + 1e-3)
        )

    if {"no2", "o3"}.issubset(df.columns):
        poll_cols["no2_o3_ratio"] = (
            df["no2"].shift(1) / (df["o3"].shift(1) + 1e-3)
        ).clip(upper=100)

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
        met_cols["heat_index_proxy"] = t_l1 * h_l1 / 100.0

    # IMPR #5: Ventilation index = wind_speed × mixed_layer_proxy
    # Physical basis: higher wind + lower humidity → faster pollutant dispersal
    if ws_col_14 and temp_col and hum_col:
        t_l1 = df[temp_col].shift(1)
        h_l1 = df[hum_col].shift(1)
        ws_l1 = df[ws_col_14].shift(1)
        met_cols["ventilation_index"] = ws_l1 * (1.0 - h_l1 / 100.0) * (t_l1 + 273.15) / 300.0

    if ws_col_14 and "pm25" in df.columns:
        met_cols["wind_dispersal"]                = df[ws_col_14].shift(1) / (df["pm25"].shift(1) + 5)
        met_cols["interaction_pm25_wind_inverse"] = (
            df["pm25"].shift(1) / (df[ws_col_14].shift(1) + 0.5)
        ).clip(upper=500)

    if hum_col and "pm25" in df.columns:
        met_cols["interaction_pm25_humidity"] = (
            df["pm25"].shift(1) * df[hum_col].shift(1) / 100.0
        )

    dew_col = _resolve_col(df, "dew_point", "dew_point_2m")
    if temp_col and dew_col:
        met_cols["dew_point_depression"]     = df[temp_col].shift(1) - df[dew_col].shift(1)
        met_cols["dew_pt_depression_roll24"] = (
            (df[temp_col].shift(1) - df[dew_col].shift(1)).rolling(24, min_periods=1).mean()
        )

    # IMPR #2: Boundary-layer height proxy
    # Shallow mixing at early morning + low wind + high humidity → PM2.5 accumulation
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

        # IMPR #2 continued: explicit morning boundary-layer collapse flag
        met_cols["is_morning_stagnation"] = (
            (df["datetime"].dt.hour.isin([5, 6, 7, 8])) &
            (ws_lag1_stag < 5.0) &
            (df[hum_col].shift(1) > 60.0)
        ).astype(float)

    if "precipitation" in df.columns:
        prec = df["precipitation"].shift(1)
        met_cols["rain_24h"]   = prec.rolling(24, min_periods=1).sum()
        met_cols["rain_72h"]   = prec.rolling(72, min_periods=1).sum()
        met_cols["rain_event"] = (prec > 0.1).astype(int)

        # Monsoon washout: monsoon × rain_24h interaction
        if "is_monsoon" in df.columns:
            met_cols["monsoon_washout"] = df["is_monsoon"] * met_cols["rain_24h"]

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
    # NOTE: We intentionally do NOT drop rows here anymore.
    # Dropping rows where target_aqi_72h is NaN silently removes the most
    # recent 72 hours from the feature store, making the dashboard always
    # show "Last updated: now - 3 days".
    #
    # Instead we tag each row so downstream consumers can filter themselves:
    #   is_training_ready = True  → has both 30-day lag and 72h future target
    #   is_training_ready = False → recent rows (last ~72h): use for inference only
    required_non_null = ["aqi_same_hour_30days_ago", "target_aqi_72h"]
    df["is_training_ready"] = df[required_non_null].notna().all(axis=1)
    n_training = df["is_training_ready"].sum()
    n_inference_only = (~df["is_training_ready"]).sum()
    print(f" -> Tagged {n_training:,} training-ready rows and "
          f"{n_inference_only:,} inference-only rows (recent, no future target yet).")

    assert df["datetime"].is_monotonic_increasing, "CRITICAL: Temporal order broken."

    # ── STEP 16: DTYPE SAFETY PASS ───────────────────────────────────────────
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    non_date_obj = [c for c in obj_cols if c != "datetime"]
    if non_date_obj:
        print(f" -> WARNING: Coercing {len(non_date_obj)} non-numeric columns to float64: {non_date_obj}")
        for col in non_date_obj:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64).fillna(0.0)

    feature_cols = [c for c in df.columns if c not in
                    ["datetime", "timestamp"] and not c.startswith("target_")]
    print(f" -> Feature matrix: {len(feature_cols)} engineered features, {len(df):,} rows.")
    return df


# ── 3. Production Pipeline Entrypoint ─────────────────────────────────────────

def _cleanup_old_fsh_images(db) -> int:
    """
    Delete all documents from the fsh_images collection (SHAP/feature importance
    plots saved as binary blobs) before uploading new ones. This prevents
    unbounded growth on the free MongoDB cluster.

    Returns the count of deleted documents.
    """
    try:
        result = db["fsh_images"].delete_many({})
        n = result.deleted_count
        if n > 0:
            print(f"  [FSH cleanup] Deleted {n} old FSH image document(s) from MongoDB.")
        return n
    except Exception as e:
        print(f"  [FSH cleanup] WARNING: Could not clean fsh_images: {e}")
        return 0


def process_all(db=None):
    """
    Build the engineered feature matrix from karachi_aqi_dataset and upsert
    results into processed_features.

    - All data exclusively read from and written to MongoDB.
    - No local files created at any point.
    - Cleans old FSH image blobs before each run to save free-tier storage.

    Parameters
    ----------
    db : pymongo.database.Database, optional
        Pass an existing authenticated db handle to reuse a connection.
        If None, a new MongoClient is created from MONGODB_URI env var.
    """
    _owns_client = db is None
    client       = None

    if _owns_client:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("CRITICAL: MONGODB_URI missing from environment.")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
        db     = client["karachi_aqi"]

    try:
        # Clean up large FSH image blobs from the previous run
        _cleanup_old_fsh_images(db)

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

        mongo_df = processed_df.copy()
        mongo_df["timestamp"] = mongo_df["datetime"].apply(lambda x: x.to_pydatetime())
        mongo_df["datetime"]  = mongo_df["datetime"].apply(lambda x: x.to_pydatetime())

        features_payload = mongo_df.to_dict(orient="records")

        clean_payload = [
            {k: (None if isinstance(v, float) and pd.isna(v) else v)
             for k, v in doc.items()}
            for doc in features_payload
        ]

        print(f"Prepared {len(clean_payload):,} documents.")

        def _bulk_upsert(collection, payload):
            collection.create_index("datetime", unique=True)
            ops = [
                UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True)
                for r in payload
            ]
            total = (len(ops) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE
            for i in range(0, len(ops), BULK_BATCH_SIZE):
                batch_num = i // BULK_BATCH_SIZE + 1
                result = collection.bulk_write(ops[i: i + BULK_BATCH_SIZE], ordered=False)
                print(f"  Batch {batch_num}/{total} — "
                      f"upserted: {result.upserted_count}, modified: {result.modified_count}")

        # Write ALL rows including recent ones without future targets.
        # Dashboard + inference reads here — must include the latest hours.
        print(f"\nWriting {len(clean_payload):,} rows to 'processed_features' (dashboard + inference)...")
        _bulk_upsert(db["processed_features"], clean_payload)

        # Write only training-ready rows to a separate collection.
        # Model trainer reads here — only rows with valid 72h future targets.
        training_payload = [r for r in clean_payload if r.get("is_training_ready")]
        print(f"Writing {len(training_payload):,} rows to 'processed_features_training' (model training)...")
        _bulk_upsert(db["processed_features_training"], training_payload)

        print("\nFeature store synchronised successfully.")
        print("=" * 70 + "\n")

    finally:
        if _owns_client and client is not None:
            client.close()


if __name__ == "__main__":
    process_all()