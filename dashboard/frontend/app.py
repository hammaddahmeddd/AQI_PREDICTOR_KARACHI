"""
app.py — Karachi AQI Forecast Dashboard (Streamlit)
Deploy on Streamlit Community Cloud.

Set BACKEND_URL in Streamlit secrets or as environment variable.
"""

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Production backend URL on Render.
# You can still override this in Streamlit Cloud secrets with:
# BACKEND_URL = "https://aqi-predictor-karachi.onrender.com"
DEFAULT_BACKEND_URL = "https://aqi-predictor-karachi.onrender.com"


def get_backend_url() -> str:
    """Get backend URL from environment/secrets, with Render URL as safe default."""
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
    if aqi is None or np.isnan(aqi):
        return "#64748b"
    for lo, hi, color, _ in AQI_BANDS:
        if lo <= aqi <= hi:
            return color
    return "#7e0023"


def aqi_label(aqi: float) -> str:
    if aqi is None or np.isnan(aqi):
        return "Unknown"
    for lo, hi, _, label in AQI_BANDS:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


def metric_color(value: float, metric: str) -> str:
    """Color code a metric based on whether lower/higher is better."""
    if value is None:
        return "normal"
    if metric in ("mae", "rmse", "mape"):
        if metric == "r2":
            return "normal"
        return "inverse"  # lower is better
    return "normal"


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

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_forecast, tab_compare, tab_features, tab_pipeline = st.tabs([
    "📊 Overview",
    "🔮 Forecast",
    "⚖️ Model Comparison",
    "🔬 Feature Importance",
    "⚙️ Pipeline Status",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    latest = fetch("/api/latest-aqi")
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

            # 1h, 6h, 24h lag comparisons
            aqi_1h = latest.get("aqi_lag_1")
            aqi_6h = latest.get("aqi_lag_6")
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

    # ── AQI History chart ─────────────────────────────────────────────────────
    st.markdown(f"#### AQI History — Last {history_hours} hours")

    if history_data and history_data.get("history"):
        hdf = pd.DataFrame(history_data["history"])
        hdf["time"] = pd.to_datetime(hdf.get("timestamp", hdf.get("datetime")))
        hdf = hdf.sort_values("time")

        fig = go.Figure()

        # AQI band shading
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

        # PM2.5 alongside
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
        c1.metric("MAE", fmt(m.get("test_mae"), 1))
        c2.metric("RMSE", fmt(m.get("test_rmse"), 1))
        c3.metric("R²", fmt(m.get("test_r2"), 3))
        c4.metric("Coverage", pct(m.get("conformal_global_coverage")))
    else:
        st.info("Train the model to see metrics here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast
# ══════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    st.markdown(f"#### {selected_model_label} Predictions — {selected_horizon_label} horizon")

    pred_data = fetch(f"/api/predictions/{selected_model}/{selected_horizon}?limit=336")

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
# TAB 3 — Model Comparison (appears on every tab per requirement)
# ══════════════════════════════════════════════════════════════════════════════

def render_comparison_section(selected_horizon_val: int):
    """Renders the cross-model comparison table + charts for a given horizon."""
    comparison = fetch("/api/model-comparison")
    if not comparison or not comparison.get("rows"):
        st.info("No model metrics available. Run the training pipeline first.")
        return

    rows = comparison["rows"]
    df_all = pd.DataFrame(rows)

    # Filter to current horizon
    df_h = df_all[df_all["horizon"] == selected_horizon_val].copy()

    if df_h.empty:
        st.info(f"No data for {selected_horizon_val}h horizon yet.")
        return

    # ── Metric table ──────────────────────────────────────────────────────────
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

    # Percent columns
    for col in ["Coverage ↑", "Cov>150 ↑", "Cov>200 ↑"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda v: f"{v*100:.1f}%" if v is not None else "—")

    # Format floats
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

    # ── Bar charts ────────────────────────────────────────────────────────────
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

    # ── All horizons radar / grouped bar ──────────────────────────────────────
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
    st.markdown(f"#### SHAP Feature Importance — {selected_model_label} · {selected_horizon_label}")

    # Model comparison strip (per requirement: comparison on every page)
    st.markdown("---")
    st.markdown(f"#### Model Comparison · {selected_horizon_label}")
    render_comparison_section(selected_horizon)
    st.markdown("---")

    feat_data = fetch(f"/api/top-features/{selected_model}/{selected_horizon}")

    if feat_data and feat_data.get("top_features"):
        feats = feat_data["top_features"]
        fdf = pd.DataFrame(feats)

        if "mean_abs_shap" in fdf.columns and "feature" in fdf.columns:
            fdf = fdf.nlargest(20, "mean_abs_shap")
            fig_feat = go.Figure(go.Bar(
                x=fdf["mean_abs_shap"],
                y=fdf["feature"],
                orientation="h",
                marker=dict(
                    color=fdf["mean_abs_shap"],
                    colorscale=[[0, "#1e2a42"], [1, "#38bdf8"]],
                    showscale=False,
                ),
            ))
            feat_theme = plotly_theme()
            feat_theme["yaxis"] = {
                **feat_theme.get("yaxis", {}),
                "autorange": "reversed",
            }

            fig_feat.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Mean |SHAP value|",
                **feat_theme,
            )
            st.plotly_chart(fig_feat, use_container_width=True)

            with st.expander("📋 Raw feature importance data"):
                st.dataframe(fdf.style.format({"mean_abs_shap": "{:.4f}"}), use_container_width=True)
        else:
            st.info("Feature data is available but in an unexpected format.")
            st.json(feats[:5])
    else:
        st.info("No SHAP feature importance data available yet. Train the model to generate it.")

    meta_cols = st.columns(2)
    if feat_data:
        meta_cols[0].caption(f"Total features: {feat_data.get('n_features', '—')}")
        meta_cols[1].caption(f"Trained at: {feat_data.get('trained_at', '—')[:19] if feat_data.get('trained_at') else '—'}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Pipeline Status
# ══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.markdown("#### Pipeline Status")

    # Model comparison strip (per requirement)
    st.markdown("---")
    st.markdown(f"#### Model Comparison · {selected_horizon_label}")
    render_comparison_section(selected_horizon)
    st.markdown("---")

    status_data = fetch("/api/pipeline-status?limit=30")

    if status_data and status_data.get("statuses"):
        sdf = pd.DataFrame(status_data["statuses"])

        # Summary chips
        if "status" in sdf.columns:
            success_count = (sdf["status"] == "SUCCESS").sum()
            fail_count = (sdf["status"] == "FAILED").sum()

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total runs (last 30)", len(sdf))
            sc2.metric("✅ Success", success_count)
            sc3.metric("❌ Failed", fail_count)

        # Color-coded status table
        def color_status(val):
            if val == "SUCCESS":
                return "color: #22c55e"
            if val == "FAILED":
                return "color: #ef4444"
            return "color: #f59e0b"

        disp_cols = [c for c in ["pipeline", "step", "status", "logged_at", "error"] if c in sdf.columns]
        styled = sdf[disp_cols].style.applymap(color_status, subset=["status"])
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.info("No pipeline status records found. Run the feature or training pipeline.")

    st.divider()
    health = fetch("/")
    if health:
        st.success(f"Backend healthy · {health.get('timestamp', '')[:19]} UTC")
    else:
        st.error("Backend unreachable")