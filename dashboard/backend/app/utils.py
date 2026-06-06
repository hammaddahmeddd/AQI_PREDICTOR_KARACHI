from datetime import datetime
from math import isnan, isinf
from typing import Any

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
}

HORIZONS = [24, 48, 72]

def clean_value(value: Any):
    if isinstance(value, float) and (isnan(value) or isinf(value)):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value

def clean_doc(doc: dict | None) -> dict | None:
    if not doc:
        return None
    return {k: clean_value(v) for k, v in doc.items() if k != "_id"}

def aqi_category(aqi: float | int | None) -> dict:
    if aqi is None:
        return {"label": "Unknown", "tone": "muted", "advice": "Waiting for latest AQI data."}
    if aqi <= 50:
        return {"label": "Good", "tone": "good", "advice": "Air quality is satisfactory."}
    if aqi <= 100:
        return {"label": "Moderate", "tone": "moderate", "advice": "Sensitive groups should consider limiting long outdoor activity."}
    if aqi <= 150:
        return {"label": "Unhealthy for Sensitive Groups", "tone": "usg", "advice": "Sensitive groups should limit outdoor activity. N95 masks recommended."}
    if aqi <= 200:
        return {"label": "Unhealthy", "tone": "unhealthy", "advice": "Reduce outdoor activity and consider wearing a mask outside."}
    if aqi <= 300:
        return {"label": "Very Unhealthy", "tone": "very_unhealthy", "advice": "Avoid strenuous outdoor activity."}
    return {"label": "Hazardous", "tone": "hazardous", "advice": "Stay indoors and keep windows closed."}

def safe_round(value: Any, digits: int = 1):
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except Exception:
        return None

def model_display_name(model_name: str):
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())
