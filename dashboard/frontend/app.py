"""
app.py — Karachi AQI Forecast Dashboard (Streamlit)
Deploy on Streamlit Community Cloud.

Set BACKEND_URL in Streamlit secrets or as environment variable.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Production backend URL on Render.
DEFAULT_BACKEND_URL = "https://aqi-predictor-karachi.onrender.com"

ALERTS_STORAGE_KEY = "aqi_alerts_config"


def get_backend_url() -> str:
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = st.secrets.get("BACKEND_URL")
        if secret_url:
            return str(secret_url).rstrip("/")
    except Exception:
        pass
    return DEFAULT_BACKEND_URL.rstrip("/")


BACKEND_URL = get_backend_url()

MODEL_OPTIONS = {
    "Random Forest": "random_forest",
    "XGBoost":       "xgboost",
    "Ridge":         "ridge",
}

HORIZON_OPTIONS = {"24 hours": 24, "48 hours": 48, "72 hours": 72}

AQI_BANDS = [
    (0,   50,  "#00e400", "Good"),
    (51,  100, "#ffff00", "Moderate"),
    (101, 150, "#ff7e00", "Unhealthy for Sensitive Groups"),
    (151, 200, "#ff0000", "Unhealthy"),
    (201, 300, "#8f3f97", "Very Unhealthy"),
    (301, 500, "#7e0023", "Hazardous"),
]

# Alert threshold definitions
ALERT_LEVELS = {
    "Good (>50)":                    50,
    "Moderate (>100)":              100,
    "Unhealthy for Sensitive (>150)": 150,
    "Unhealthy (>200)":             200,
    "Very Unhealthy (>300)":        300,
    "Hazardous (>400)":             400,
}

# ── Session state defaults ─────────────────────────────────────────────────────

if "alert_threshold" not in st.session_state:
    st.session_state["alert_threshold"] = 150
if "alert_enabled" not in st.session_state:
    st.session_state["alert_enabled"] = True
if "alert_email" not in st.session_state:
    st.session_state["alert_email"] = ""
if "alert_sound" not in st.session_state:
    st.session_state["alert_sound"] = True
if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = []

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .metric-label {
    font-family: 'Space Mono', monospace;
}

.main { background: #0a0e1a; }

section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2a42;
}

.stMetric {
    background: #111827;
    border: 1px solid #1e2a42;
    border-radius: 12px;
    padding: 16px;
}

.stMetric label {
    color: #64748b !important;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stMetric [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace;
    font-size: 28px;
}

.stMetric [data-testid="stMetricDelta"] {
    font-size: 13px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1e2a42;
    border-radius: 12px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #64748b;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8;
    border-bottom-color: #38bdf8;
}

.aqi-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.alert-card {
    background: #1a0a0a;
    border: 1px solid #7e0023;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.alert-card-warn {
    background: #1a130a;
    border: 1px solid #ff7e00;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.alert-card-info {
    background: #0a141a;
    border: 1px solid #38bdf8;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.block-container { padding-top: 1.5rem; }

.stAlert { border-radius: 10px; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=(10, 60))
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        st.warning(
            f"Backend is taking longer than expected for `{endpoint}`. "
            "If this is the first load, wait 30 seconds and click Refresh data."
        )
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to backend. Make sure the Render backend service is running."
        )
        return None
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return None


def aqi_color(aqi: float) -> str:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "#64748b"
    for lo, hi, color, _ in AQI_BANDS:
        if lo <= aqi <= hi:
            return color
    return "#7e0023"


def aqi_label(aqi: float) -> str:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown"
    for lo, hi, _, label in AQI_BANDS:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


def fmt(v, decimals=2, suffix="") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def plotly_theme() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis=dict(gridcolor="#1e2a42", zeroline=False, color="#94a3b8"),
        yaxis=dict(gridcolor="#1e2a42", zeroline=False, color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a42"),
    )


def check_and_fire_alerts(current_aqi: float | None, predicted_values: list[float]) -> list[dict]:
    """
    Check current AQI and predicted values against the configured threshold.
    Returns a list of alert dicts that were triggered.
    """
    if not st.session_state["alert_enabled"]:
        return []

    threshold = st.session_state["alert_threshold"]
    fired = []

    if current_aqi is not None and current_aqi > threshold:
        fired.append({
            "type": "CURRENT",
            "aqi": current_aqi,
            "threshold": threshold,
            "label": aqi_label(current_aqi),
            "color": aqi_color(current_aqi),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "message": f"Current AQI ({current_aqi:.0f}) exceeds threshold ({threshold})"
        })

    for i, val in enumerate(predicted_values):
        if val is not None and val > threshold:
            fired.append({
                "type": "FORECAST",
                "aqi": val,
                "threshold": threshold,
                "label": aqi_label(val),
                "color": aqi_color(val),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "message": f"Forecast AQI ({val:.0f}) at +{i+1}h exceeds threshold ({threshold})"
            })
            break  # only alert on first forecast breach

    # Append new unique alerts to log (avoid duplicates per session)
    for alert in fired:
        existing_msgs = [a["message"] for a in st.session_state["alert_log"]]
        if alert["message"] not in existing_msgs:
            st.session_state["alert_log"].insert(0, alert)

    # Keep log to last 50 entries
    st.session_state["alert_log"] = st.session_state["alert_log"][:50]

    return fired


def render_alert_banner(fired_alerts: list[dict]):
    """Show prominent banner alerts at top of page if any alerts fired."""
    for alert in fired_alerts:
        icon = "🚨" if alert["aqi"] > 200 else "⚠️"
        color = alert["color"]
        st.markdown(
            f"""
            <div style='background:{color}22;border:2px solid {color};border-radius:12px;
                        padding:14px 20px;margin-bottom:10px;display:flex;align-items:center;gap:12px'>
              <span style='font-size:28px'>{icon}</span>
              <div>
                <div style='font-family:Space Mono;font-size:13px;font-weight:700;color:{color}'>
                  {alert["type"]} AQI ALERT — {alert["label"].upper()}
                </div>
                <div style='font-family:DM Sans;font-size:14px;color:#e2e8f0;margin-top:4px'>
                  {alert["message"]}
                </div>
                <div style='font-family:Space Mono;font-size:11px;color:#64748b;margin-top:4px'>
                  Triggered at {alert["timestamp"]}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌫️ Karachi AQI")
    st.markdown(
        "<p style='color:#64748b;font-size:13px;font-family:Space Mono'>Real-time air quality forecasting powered by ML</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Model Selection")
    selected_model_label = st.radio(
        "Forecast model",
        list(MODEL_OPTIONS.keys()),
        index=1,  # Default: XGBoost
        label_visibility="collapsed",
    )
    selected_model = MODEL_OPTIONS[selected_model_label]

    st.markdown("### Forecast Horizon")
    selected_horizon_label = st.radio(
        "Horizon",
        list(HORIZON_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_horizon = HORIZON_OPTIONS[selected_horizon_label]

    st.divider()

    # ── Alerts quick-config in sidebar ────────────────────────────────────────
    st.markdown("### 🔔 Alert Settings")
    st.session_state["alert_enabled"] = st.toggle(
        "Enable AQI Alerts",
        value=st.session_state["alert_enabled"],
    )

    st.session_state["alert_threshold"] = st.slider(
        "Alert threshold (AQI)",
        min_value=50,
        max_value=400,
        value=st.session_state["alert_threshold"],
        step=10,
        help="Receive an alert when AQI exceeds this level",
        disabled=not st.session_state["alert_enabled"],
    )

    threshold_val = st.session_state["alert_threshold"]
    threshold_color = aqi_color(threshold_val + 1)
    threshold_label = aqi_label(threshold_val + 1)
    st.markdown(
        f"<p style='font-size:12px;font-family:Space Mono;color:{threshold_color}'>"
        f"🎯 Alerting at: {threshold_label}</p>",
        unsafe_allow_html=True,
    )

    if st.session_state["alert_log"]:
        st.markdown(
            f"<p style='font-size:12px;color:#f59e0b'>⚠️ {len(st.session_state['alert_log'])} alert(s) in log</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### History Window")
    history_hours = st.slider("Hours of AQI history", 24, 720, 168, step=24,
                               help="For the AQI history chart on the Overview tab")

    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    f"<h1 style='color:#e2e8f0;margin-bottom:0'>Karachi AQI Forecast</h1>"
    f"<p style='color:#64748b;font-family:Space Mono;font-size:13px;margin-top:4px'>"
    f"Model: <span style='color:#38bdf8'>{selected_model_label}</span> · "
    f"Horizon: <span style='color:#38bdf8'>{selected_horizon_label}</span>"
    f"</p>",
    unsafe_allow_html=True,
)

# ── Pre-fetch data for alert evaluation ───────────────────────────────────────

_latest_data = fetch("/api/latest-aqi")
_pred_data    = fetch(f"/api/predictions/{selected_model}/{selected_horizon}?limit=336")

_current_aqi = None
_pred_values  = []

if _latest_data:
    _current_aqi = _latest_data.get("aqi")

if _pred_data and _pred_data.get("predictions"):
    _pdf = pd.DataFrame(_pred_data["predictions"])
    if "predicted" in _pdf.columns:
        _pred_values = _pdf["predicted"].dropna().tolist()[:selected_horizon]

# Evaluate alerts
_fired_alerts = check_and_fire_alerts(_current_aqi, _pred_values)

# Show banner alerts at the very top (below header)
if _fired_alerts:
    render_alert_banner(_fired_alerts)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_forecast, tab_compare, tab_features, tab_alerts, tab_pipeline = st.tabs([
    "📊 Overview",
    "🔮 Forecast",
    "⚖️ Model Comparison",
    "🔬 Feature Importance",
    "🔔 Alerts",
    "⚙️ Pipeline Status",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    latest = _latest_data
    history_data = fetch(f"/api/aqi-history?hours={history_hours}")

    # ── Current AQI banner ────────────────────────────────────────────────────
    if latest:
        current_aqi = latest.get("aqi")
        color = aqi_color(current_aqi)
        label = aqi_label(current_aqi)

        col_aqi, col_meta = st.columns([1, 2])

        with col_aqi:
            st.markdown(
                f"""
                <div style='background:{color}18;border:2px solid {color};border-radius:16px;
                            padding:28px;text-align:center;margin-bottom:12px'>
                  <div style='font-family:Space Mono;font-size:11px;color:{color};
                              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>
                    Current AQI
                  </div>
                  <div style='font-family:Space Mono;font-size:64px;font-weight:700;
                              color:{color};line-height:1'>{fmt(current_aqi, 0)}</div>
                  <div style='font-family:DM Sans;font-size:14px;color:#94a3b8;
                              margin-top:10px'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_meta:
            ts = latest.get("timestamp") or latest.get("datetime", "")
            st.markdown(f"<p style='color:#64748b;font-size:12px;font-family:Space Mono'>Last updated: {ts[:19].replace('T',' ')} UTC</p>", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("PM2.5", fmt(latest.get("pm25"), 1, " µg/m³"))
            r2.metric("PM10",  fmt(latest.get("pm10"), 1, " µg/m³"))
            r3.metric("NO₂",   fmt(latest.get("no2"),  1, " µg/m³"))

            r4, r5, r6 = st.columns(3)
            r4.metric("Temperature", fmt(latest.get("temperature"), 1, " °C"))
            r5.metric("Humidity",    fmt(latest.get("humidity"), 0, "%"))
            r6.metric("Wind Speed",  fmt(latest.get("wind_speed"), 1, " km/h"))

            aqi_1h  = latest.get("aqi_lag_1")
            aqi_6h  = latest.get("aqi_lag_6")
            aqi_24h = latest.get("aqi_lag_24")

            r7, r8, r9 = st.columns(3)
            if current_aqi and aqi_1h:
                r7.metric("AQI 1h ago", fmt(aqi_1h, 0), delta=fmt(current_aqi - aqi_1h, 1))
            if current_aqi and aqi_6h:
                r8.metric("AQI 6h ago", fmt(aqi_6h, 0), delta=fmt(current_aqi - aqi_6h, 1))
            if current_aqi and aqi_24h:
                r9.metric("AQI 24h ago", fmt(aqi_24h, 0), delta=fmt(current_aqi - aqi_24h, 1))
    else:
        st.warning("Could not load current AQI from backend.")

    st.divider()

    # ── AQI Band reference ────────────────────────────────────────────────────
    st.markdown("#### AQI Scale Reference")
    band_cols = st.columns(len(AQI_BANDS))
    for col, (lo, hi, color, label) in zip(band_cols, AQI_BANDS):
        col.markdown(
            f"<div style='background:{color}22;border:1px solid {color};border-radius:8px;"
            f"padding:8px;text-align:center'>"
            f"<div style='font-family:Space Mono;font-size:10px;color:{color};font-weight:700'>{lo}–{hi}</div>"
            f"<div style='font-family:DM Sans;font-size:11px;color:#94a3b8;margin-top:4px'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── AQI History chart ─────────────────────────────────────────────────────
    st.markdown(f"#### AQI History — Last {history_hours} hours")

    if history_data and history_data.get("history"):
        hdf = pd.DataFrame(history_data["history"])
        hdf["time"] = pd.to_datetime(hdf.get("timestamp", hdf.get("datetime")))
        hdf = hdf.sort_values("time")

        fig = go.Figure()

        band_colors = [
            "rgba(0, 228, 0, 0.08)",
            "rgba(255, 255, 0, 0.08)",
            "rgba(255, 126, 0, 0.08)",
            "rgba(255, 0, 0, 0.08)",
            "rgba(143, 63, 151, 0.08)",
            "rgba(126, 0, 35, 0.08)",
        ]
        band_bounds = [0, 50, 100, 150, 200, 300, 500]

        for i, (lo, hi) in enumerate(zip(band_bounds, band_bounds[1:])):
            fig.add_hrect(y0=lo, y1=hi, fillcolor=band_colors[i], line_width=0)

        # Alert threshold line
        if st.session_state["alert_enabled"]:
            thr = st.session_state["alert_threshold"]
            fig.add_hline(
                y=thr,
                line_color="#f59e0b",
                line_dash="dash",
                line_width=1.5,
                annotation_text=f"Alert threshold ({thr})",
                annotation_position="top right",
                annotation_font_color="#f59e0b",
                annotation_font_size=11,
            )

        fig.add_trace(go.Scatter(
            x=hdf["time"],
            y=hdf["aqi"],
            mode="lines",
            name="AQI",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.06)",
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            yaxis_title="AQI",
            **plotly_theme(),
        )

        st.plotly_chart(fig, use_container_width=True)

        if "pm25" in hdf.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hdf["time"], y=hdf["pm25"],
                mode="lines", name="PM2.5",
                line=dict(color="#f472b6", width=2),
            ))
            fig2.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                yaxis_title="PM2.5 (µg/m³)",
                **plotly_theme(),
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No history data available yet.")

    # ── Quick model metrics strip ──────────────────────────────────────────────
    st.divider()
    st.markdown(f"#### {selected_model_label} · {selected_horizon_label} Metrics")

    metrics_data = fetch(f"/api/metrics/{selected_horizon}")
    if metrics_data and metrics_data.get("models", {}).get(selected_model):
        m = metrics_data["models"][selected_model]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",      fmt(m.get("test_mae"), 1))
        c2.metric("RMSE",     fmt(m.get("test_rmse"), 1))
        c3.metric("R²",       fmt(m.get("test_r2"), 3))
        c4.metric("Coverage", pct(m.get("conformal_global_coverage")))
    else:
        st.info("Train the model to see metrics here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast
# ══════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    st.markdown(f"#### {selected_model_label} Predictions — {selected_horizon_label} horizon")

    pred_data = _pred_data

    if pred_data and pred_data.get("predictions"):
        pdf = pd.DataFrame(pred_data["predictions"])
        pdf["time"] = pd.to_datetime(pdf.get("timestamp", pdf.get("datetime", pdf.index)))
        pdf = pdf.sort_values("time")

        # ── Model metrics strip ───────────────────────────────────────────────
        metrics_data2 = fetch(f"/api/metrics/{selected_horizon}")
        if metrics_data2 and metrics_data2.get("models", {}).get(selected_model):
            m = metrics_data2["models"][selected_model]
            cols = st.columns(6)
            cols[0].metric("MAE",      fmt(m.get("test_mae"), 1))
            cols[1].metric("RMSE",     fmt(m.get("test_rmse"), 1))
            cols[2].metric("MAPE",     fmt(m.get("test_mape"), 2, "%"))
            cols[3].metric("R²",       fmt(m.get("test_r2"), 3))
            cols[4].metric("Coverage", pct(m.get("conformal_global_coverage")))
            cols[5].metric("Skill",    fmt(m.get("forecast_skill_score"), 3))

        st.markdown("")

        fig = go.Figure()

        # Alert threshold line on forecast chart
        if st.session_state["alert_enabled"]:
            thr = st.session_state["alert_threshold"]
            fig.add_hline(
                y=thr,
                line_color="#f59e0b",
                line_dash="dash",
                line_width=1.5,
                annotation_text=f"Alert ({thr})",
                annotation_position="top right",
                annotation_font_color="#f59e0b",
                annotation_font_size=11,
            )

        # Confidence band
        if "pi_upper" in pdf.columns and "pi_lower" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([pdf["time"], pdf["time"].iloc[::-1]]),
                y=pd.concat([pdf["pi_upper"], pdf["pi_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(56,189,248,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Conformal PI (95%)",
                showlegend=True,
            ))

        # Actual
        if "actual" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["time"], y=pdf["actual"],
                mode="lines",
                name="Actual AQI",
                line=dict(color="#94a3b8", width=1.5),
            ))

        # Predicted
        fig.add_trace(go.Scatter(
            x=pdf["time"], y=pdf["predicted"],
            mode="lines",
            name=f"Predicted ({selected_model_label})",
            line=dict(color="#38bdf8", width=2.5),
        ))

        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="AQI",
            hovermode="x unified",
            **plotly_theme(),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Residuals plot ────────────────────────────────────────────────────
        if "actual" in pdf.columns and "predicted" in pdf.columns:
            pdf["residual"] = pdf["actual"] - pdf["predicted"]
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("##### Residuals over time")
                fig_res = go.Figure()
                fig_res.add_hline(y=0, line_color="#1e2a42")
                fig_res.add_trace(go.Scatter(
                    x=pdf["time"], y=pdf["residual"],
                    mode="lines",
                    line=dict(color="#f472b6", width=1.5),
                    name="Residual",
                ))
                fig_res.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                    yaxis_title="Actual − Predicted",
                    **plotly_theme(),
                )
                st.plotly_chart(fig_res, use_container_width=True)

            with col_b:
                st.markdown("##### Actual vs Predicted")
                fig_scatter = go.Figure()
                rng = [
                    min(pdf["actual"].min(), pdf["predicted"].min()),
                    max(pdf["actual"].max(), pdf["predicted"].max()),
                ]
                fig_scatter.add_trace(go.Scatter(
                    x=[rng[0], rng[1]], y=[rng[0], rng[1]],
                    mode="lines",
                    line=dict(color="#1e2a42", dash="dash"),
                    name="Perfect",
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=pdf["actual"], y=pdf["predicted"],
                    mode="markers",
                    marker=dict(color="#38bdf8", size=4, opacity=0.6),
                    name="Predictions",
                ))
                fig_scatter.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    **plotly_theme(),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Raw data table ────────────────────────────────────────────────────
        with st.expander("📋 Raw prediction data"):
            disp = pdf.copy()
            disp["time"] = disp["time"].dt.strftime("%Y-%m-%d %H:%M")
            num_cols = [c for c in ["actual", "predicted", "pi_lower", "pi_upper", "residual"] if c in disp.columns]
            st.dataframe(
                disp[["time"] + num_cols].style.format({c: "{:.1f}" for c in num_cols}),
                use_container_width=True,
                height=300,
            )
    else:
        st.info("No predictions found. Run the training pipeline to generate predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

def render_comparison_section(selected_horizon_val: int):
    """Renders the cross-model comparison table + charts for a given horizon."""
    comparison = fetch("/api/model-comparison")
    if not comparison or not comparison.get("rows"):
        st.info("No model metrics available. Run the training pipeline first.")
        return

    rows = comparison["rows"]
    df_all = pd.DataFrame(rows)

    df_h = df_all[df_all["horizon"] == selected_horizon_val].copy()

    if df_h.empty:
        st.info(f"No data for {selected_horizon_val}h horizon yet.")
        return

    display_cols = {
        "display_name": "Model",
        "mae":           "MAE ↓",
        "rmse":          "RMSE ↓",
        "mape":          "MAPE % ↓",
        "r2":            "R² ↑",
        "coverage":      "Coverage ↑",
        "coverage_gt150":"Cov>150 ↑",
        "coverage_gt200":"Cov>200 ↑",
        "skill":         "Skill ↑",
        "cv_rmse":       "CV RMSE ↓",
    }

    disp = df_h[list(display_cols.keys())].rename(columns=display_cols).copy()

    for col in ["Coverage ↑", "Cov>150 ↑", "Cov>200 ↑"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda v: f"{v*100:.1f}%" if v is not None else "—")

    for col in ["MAE ↓", "RMSE ↓", "CV RMSE ↓"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda v: f"{v:.1f}" if v is not None else "—")

    for col in ["MAPE % ↓"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda v: f"{v:.2f}%" if v is not None else "—")

    for col in ["R² ↑", "Skill ↑"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda v: f"{v:.4f}" if v is not None else "—")

    st.dataframe(disp.set_index("Model"), use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### MAE by model (lower is better)")
        fig_mae = go.Figure(go.Bar(
            x=df_h["display_name"],
            y=df_h["mae"],
            marker_color=["#38bdf8", "#f472b6", "#a78bfa"],
            text=df_h["mae"].apply(lambda v: f"{v:.1f}" if v else ""),
            textposition="outside",
        ))
        fig_mae.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, yaxis_title="MAE",
            **plotly_theme(),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_right:
        st.markdown("##### R² by model (higher is better)")
        fig_r2 = go.Figure(go.Bar(
            x=df_h["display_name"],
            y=df_h["r2"],
            marker_color=["#38bdf8", "#f472b6", "#a78bfa"],
            text=df_h["r2"].apply(lambda v: f"{v:.3f}" if v else ""),
            textposition="outside",
        ))
        fig_r2.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, yaxis_title="R²",
            **plotly_theme(),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("##### MAE across all horizons")
    fig_hbar = go.Figure()
    colors = {"random_forest": "#38bdf8", "ridge": "#a78bfa", "xgboost": "#f472b6"}
    for model_key, display in [("random_forest","Random Forest"), ("ridge","Ridge"), ("xgboost","XGBoost")]:
        model_rows = df_all[df_all["model"] == model_key].sort_values("horizon")
        if model_rows.empty:
            continue
        fig_hbar.add_trace(go.Bar(
            name=display,
            x=model_rows["horizon_label"],
            y=model_rows["mae"],
            marker_color=colors[model_key],
        ))
    fig_hbar.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="MAE",
        **plotly_theme(),
    )
    st.plotly_chart(fig_hbar, use_container_width=True)


with tab_compare:
    st.markdown(f"#### Cross-Model Comparison — {selected_horizon_label}")
    render_comparison_section(selected_horizon)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

with tab_features:
    st.markdown(f"#### SHAP Feature Importance - {selected_model_label} · {selected_horizon_label}")

    feat_data = fetch(f"/api/top-features/{selected_model}/{selected_horizon}")

    if feat_data and feat_data.get("top_features"):
        feats = feat_data["top_features"]
        fdf = pd.DataFrame(feats)

        if "mean_abs_shap" in fdf.columns and "feature" in fdf.columns:
            fdf["mean_abs_shap"] = pd.to_numeric(fdf["mean_abs_shap"], errors="coerce")

            fdf = (
                fdf.dropna(subset=["mean_abs_shap"])
                   .nlargest(20, "mean_abs_shap")
                   .sort_values("mean_abs_shap", ascending=True)
            )

            fig_feat = go.Figure(go.Bar(
                x=fdf["mean_abs_shap"],
                y=fdf["feature"],
                orientation="h",
                marker=dict(color="#38bdf8"),
                text=fdf["mean_abs_shap"].apply(lambda v: f"{v:.4f}"),
                textposition="outside",
            ))

            fig_feat.update_layout(
                height=620,
                margin=dict(l=20, r=40, t=20, b=20),
                xaxis_title="Mean absolute SHAP value",
                yaxis_title="Feature",
                showlegend=False,
                **plotly_theme(),
            )

            st.plotly_chart(fig_feat, use_container_width=True)

            meta_cols = st.columns(2)
            meta_cols[0].caption(f"Total features: {feat_data.get('n_features', '—')}")
            meta_cols[1].caption(
                f"Trained at: {feat_data.get('trained_at', '—')[:19] if feat_data.get('trained_at') else '—'}"
            )

            with st.expander("Raw feature importance data"):
                st.dataframe(
                    fdf.sort_values("mean_abs_shap", ascending=False)
                       .style.format({"mean_abs_shap": "{:.6f}"}),
                    use_container_width=True
                )
        else:
            st.info("Feature data is available but in an unexpected format.")
            st.json(feats[:5])
    else:
        st.info("No SHAP feature importance data available yet. Train the model to generate it.")

    st.markdown("---")
    with st.expander("Model Comparison", expanded=False):
        render_comparison_section(selected_horizon)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Alerts (NEW)
# ══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown("#### 🔔 AQI Alert Management")
    st.markdown(
        "<p style='color:#64748b;font-size:13px;font-family:Space Mono'>"
        "Configure thresholds, view triggered alerts, and monitor hazardous AQI levels.</p>",
        unsafe_allow_html=True,
    )

    # ── Current alert status strip ────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)

    alert_status_text = "✅ Active" if st.session_state["alert_enabled"] else "⏸️ Paused"
    alert_status_color = "#22c55e" if st.session_state["alert_enabled"] else "#64748b"

    a1.markdown(
        f"<div style='background:#111827;border:1px solid #1e2a42;border-radius:12px;padding:16px;text-align:center'>"
        f"<div style='font-family:Space Mono;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Status</div>"
        f"<div style='font-family:Space Mono;font-size:20px;font-weight:700;color:{alert_status_color};margin-top:8px'>{alert_status_text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    thr_color = aqi_color(st.session_state["alert_threshold"] + 1)
    a2.markdown(
        f"<div style='background:#111827;border:1px solid #1e2a42;border-radius:12px;padding:16px;text-align:center'>"
        f"<div style='font-family:Space Mono;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Threshold</div>"
        f"<div style='font-family:Space Mono;font-size:20px;font-weight:700;color:{thr_color};margin-top:8px'>AQI {st.session_state['alert_threshold']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    current_aqi_disp = fmt(_current_aqi, 0) if _current_aqi else "—"
    current_color = aqi_color(_current_aqi) if _current_aqi else "#64748b"
    a3.markdown(
        f"<div style='background:#111827;border:1px solid #1e2a42;border-radius:12px;padding:16px;text-align:center'>"
        f"<div style='font-family:Space Mono;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Current AQI</div>"
        f"<div style='font-family:Space Mono;font-size:20px;font-weight:700;color:{current_color};margin-top:8px'>{current_aqi_disp}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    log_count = len(st.session_state["alert_log"])
    a4.markdown(
        f"<div style='background:#111827;border:1px solid #1e2a42;border-radius:12px;padding:16px;text-align:center'>"
        f"<div style='font-family:Space Mono;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Alerts (session)</div>"
        f"<div style='font-family:Space Mono;font-size:20px;font-weight:700;color:#f59e0b;margin-top:8px'>{log_count}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Alert configuration ───────────────────────────────────────────────────
    st.markdown("#### ⚙️ Alert Configuration")

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown("**Enable / Disable Alerts**")
        new_enabled = st.toggle(
            "Alerts enabled",
            value=st.session_state["alert_enabled"],
            key="alert_toggle_tab",
        )
        st.session_state["alert_enabled"] = new_enabled

        st.markdown("**Alert Threshold**")
        new_threshold = st.slider(
            "AQI threshold",
            min_value=50,
            max_value=400,
            value=st.session_state["alert_threshold"],
            step=10,
            key="alert_threshold_tab",
            help="An alert fires when AQI exceeds this value",
        )
        st.session_state["alert_threshold"] = new_threshold

        st.markdown("**Quick Presets**")
        preset_cols = st.columns(3)
        if preset_cols[0].button("Moderate\n(100)", use_container_width=True):
            st.session_state["alert_threshold"] = 100
            st.rerun()
        if preset_cols[1].button("Unhealthy\n(150)", use_container_width=True):
            st.session_state["alert_threshold"] = 150
            st.rerun()
        if preset_cols[2].button("Hazardous\n(300)", use_container_width=True):
            st.session_state["alert_threshold"] = 300
            st.rerun()

    with cfg_col2:
        st.markdown("**Alert Conditions**")
        conditions = {
            "🔴 Current AQI exceeds threshold": _current_aqi is not None and _current_aqi > st.session_state["alert_threshold"],
            "🟠 Any forecast value exceeds threshold": any(v > st.session_state["alert_threshold"] for v in _pred_values if v is not None),
            "🟡 AQI is Unhealthy for Sensitive Groups (>150)": _current_aqi is not None and _current_aqi > 150,
            "🔴 AQI is Unhealthy (>200)": _current_aqi is not None and _current_aqi > 200,
            "🟣 AQI is Very Unhealthy (>300)": _current_aqi is not None and _current_aqi > 300,
            "⚫ AQI is Hazardous (>400)": _current_aqi is not None and _current_aqi > 400,
        }

        for condition, triggered in conditions.items():
            status_icon = "🔥 TRIGGERED" if triggered else "✅ OK"
            status_color = "#ef4444" if triggered else "#22c55e"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"background:#111827;border:1px solid #1e2a42;border-radius:8px;"
                f"padding:10px 14px;margin-bottom:6px'>"
                f"<span style='font-size:13px;color:#e2e8f0'>{condition}</span>"
                f"<span style='font-family:Space Mono;font-size:11px;color:{status_color};font-weight:700'>{status_icon}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Email Notifications** *(optional / for your implementation)*")
        st.session_state["alert_email"] = st.text_input(
            "Email address",
            value=st.session_state["alert_email"],
            placeholder="you@example.com",
            help="Wire this to an email/SMS service in your backend",
        )
        if st.session_state["alert_email"]:
            st.caption("📧 Email alerts: connect to SendGrid/AWS SES in your backend service.")

    st.divider()

    # ── Forecast breach preview ───────────────────────────────────────────────
    st.markdown(f"#### 📈 {selected_horizon_label} Forecast — Alert Overlay")

    if _pred_data and _pred_data.get("predictions"):
        pdf_alert = pd.DataFrame(_pred_data["predictions"])
        pdf_alert["time"] = pd.to_datetime(
            pdf_alert.get("timestamp", pdf_alert.get("datetime", pdf_alert.index))
        )
        pdf_alert = pdf_alert.sort_values("time")

        thr = st.session_state["alert_threshold"]
        thr_color_hex = aqi_color(thr + 1)

        fig_alert = go.Figure()

        # Threshold band shading above threshold
        fig_alert.add_hrect(
            y0=thr,
            y1=max(pdf_alert["predicted"].max() * 1.1 if not pdf_alert.empty else 500, thr + 50),
            fillcolor=f"{thr_color_hex}18",
            line_width=0,
            annotation_text=f"⚠️ Alert Zone (>{thr})",
            annotation_position="top left",
            annotation_font_color=thr_color_hex,
            annotation_font_size=11,
        )

        # Threshold line
        fig_alert.add_hline(
            y=thr,
            line_color=thr_color_hex,
            line_dash="dash",
            line_width=2,
        )

        # Actual if available
        if "actual" in pdf_alert.columns:
            fig_alert.add_trace(go.Scatter(
                x=pdf_alert["time"], y=pdf_alert["actual"],
                mode="lines",
                name="Actual AQI",
                line=dict(color="#94a3b8", width=1.5),
            ))

        # Predicted — color breaches red
        breach_mask = pdf_alert["predicted"] > thr
        fig_alert.add_trace(go.Scatter(
            x=pdf_alert["time"], y=pdf_alert["predicted"],
            mode="lines+markers",
            name="Predicted AQI",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(
                color=["#ef4444" if b else "#38bdf8" for b in breach_mask],
                size=[8 if b else 4 for b in breach_mask],
            ),
        ))

        fig_alert.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="AQI",
            hovermode="x unified",
            **plotly_theme(),
        )
        st.plotly_chart(fig_alert, use_container_width=True)

        # Breach summary
        breach_count = breach_mask.sum() if not pdf_alert.empty else 0
        if breach_count > 0:
            breach_max = pdf_alert.loc[breach_mask, "predicted"].max()
            st.markdown(
                f"""
                <div class='alert-card-warn'>
                  <div style='font-family:Space Mono;font-size:13px;font-weight:700;color:#ff7e00'>
                    ⚠️ FORECAST BREACH DETECTED
                  </div>
                  <div style='font-family:DM Sans;font-size:14px;color:#e2e8f0;margin-top:6px'>
                    {breach_count} forecast hour(s) exceed your threshold of <b>{thr}</b>.
                    Peak predicted AQI: <b>{breach_max:.0f}</b> ({aqi_label(breach_max)}).
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='alert-card-info'>
                  <div style='font-family:Space Mono;font-size:13px;font-weight:700;color:#38bdf8'>
                    ✅ NO FORECAST BREACHES
                  </div>
                  <div style='font-family:DM Sans;font-size:14px;color:#e2e8f0;margin-top:6px'>
                    All {selected_horizon_label} predictions are within your threshold of {thr}.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No prediction data to overlay. Run the training pipeline first.")

    st.divider()

    # ── Alert log ─────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Alert Log (This Session)")

    col_log, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Clear log", use_container_width=True):
            st.session_state["alert_log"] = []
            st.rerun()

    if st.session_state["alert_log"]:
        for alert in st.session_state["alert_log"]:
            color = alert.get("color", "#64748b")
            icon = "🚨" if alert.get("aqi", 0) > 200 else "⚠️"
            card_class = "alert-card" if alert.get("aqi", 0) > 200 else "alert-card-warn"
            st.markdown(
                f"""
                <div class='{card_class}'>
                  <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                    <div>
                      <div style='font-family:Space Mono;font-size:12px;font-weight:700;color:{color}'>
                        {icon} [{alert.get("type","—")}] {alert.get("label","—").upper()} — AQI {alert.get("aqi",0):.0f}
                      </div>
                      <div style='font-family:DM Sans;font-size:13px;color:#e2e8f0;margin-top:4px'>
                        {alert.get("message","—")}
                      </div>
                    </div>
                    <div style='font-family:Space Mono;font-size:11px;color:#64748b;white-space:nowrap;margin-left:16px'>
                      {alert.get("timestamp","—")}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='background:#111827;border:1px solid #1e2a42;border-radius:12px;"
            "padding:32px;text-align:center;color:#64748b;font-family:Space Mono;font-size:13px'>"
            "No alerts triggered in this session.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── AQI band reference in alerts tab ─────────────────────────────────────
    st.markdown("#### 📊 AQI Health Categories & Recommended Actions")

    health_info = [
        (0,   50,  "#00e400", "Good",                             "Air quality is satisfactory. Enjoy outdoor activities."),
        (51,  100, "#ffff00", "Moderate",                         "Acceptable quality. Unusually sensitive people should consider limiting prolonged outdoor exertion."),
        (101, 150, "#ff7e00", "Unhealthy for Sensitive Groups",   "Sensitive groups (elderly, children, those with heart/lung disease) should reduce prolonged outdoor exertion."),
        (151, 200, "#ff0000", "Unhealthy",                        "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion."),
        (201, 300, "#8f3f97", "Very Unhealthy",                   "Health alert: everyone may experience more serious health effects. Avoid prolonged outdoor exertion."),
        (301, 500, "#7e0023", "Hazardous",                        "Health emergency. Everyone should avoid all outdoor exertion. Stay indoors with windows closed."),
    ]

    for lo, hi, color, label, advice in health_info:
        is_current = _current_aqi is not None and lo <= _current_aqi <= hi
        border = f"2px solid {color}" if is_current else f"1px solid {color}44"
        bg = f"{color}22" if is_current else f"{color}0a"
        marker = " ← CURRENT" if is_current else ""
        st.markdown(
            f"""
            <div style='background:{bg};border:{border};border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;display:flex;gap:16px;align-items:flex-start'>
              <div style='min-width:80px;text-align:center'>
                <div style='font-family:Space Mono;font-size:12px;font-weight:700;color:{color}'>{lo}–{hi}</div>
              </div>
              <div>
                <div style='font-family:Space Mono;font-size:13px;font-weight:700;color:{color}'>{label}{marker}</div>
                <div style='font-family:DM Sans;font-size:13px;color:#94a3b8;margin-top:4px'>{advice}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Pipeline Status
# ══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.markdown("#### Pipeline Status")

    st.markdown("---")
    st.markdown(f"#### Model Comparison · {selected_horizon_label}")
    render_comparison_section(selected_horizon)
    st.markdown("---")

    status_data = fetch("/api/pipeline-status?limit=30")

    if status_data and status_data.get("statuses"):
        sdf = pd.DataFrame(status_data["statuses"])

        if "status" in sdf.columns:
            success_count = (sdf["status"] == "SUCCESS").sum()
            fail_count    = (sdf["status"] == "FAILED").sum()

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total runs (last 30)", len(sdf))
            sc2.metric("✅ Success", success_count)
            sc3.metric("❌ Failed", fail_count)

        def color_status(val):
            if val == "SUCCESS":
                return "color: #22c55e"
            if val == "FAILED":
                return "color: #ef4444"
            return "color: #f59e0b"

        disp_cols = [c for c in ["pipeline", "step", "status", "logged_at", "error"] if c in sdf.columns]
        styled = sdf[disp_cols].style.map(color_status, subset=["status"])
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.info("No pipeline status records found. Run the feature or training pipeline.")

    st.divider()
    health = fetch("/")
    if health:
        st.success(f"Backend healthy · {health.get('timestamp', '')[:19]} UTC")
    else:
        st.error("Backend unreachable")