"""
Karachi AQI Predictor — Storage-Safe Exploratory Data Analysis (EDA) Engine
───────────────────────────────────────────────────────────────────
Data Source : MongoDB Atlas  →  karachi_aqi.processed_features (READ-ONLY)
Local Cache : eda/data/karachi_featured_cache.csv (AUTO-GITIGNORED)
Outputs     : eda/outputs/ (Statistical Profiles, Core Heatmaps, Seasonal Visuals)

STORAGE REFACTOR:
All analytical tabular summaries and evaluation data are dumped locally inside
the `eda/outputs/` folder. Absolutely zero write/insert commands are executed 
against MongoDB, completely protecting your free-tier storage quota.
"""

import os
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend safe for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Establish professional styling guidelines for environmental data plotting
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
})

# ─── PATH CONFIGURATION ──────────────────────────────────────────────────────
EDA_DIR    = Path(__file__).resolve().parent
BASE_DIR   = EDA_DIR.parent
DATA_DIR   = EDA_DIR / "data"
OUTPUT_DIR = EDA_DIR / "outputs"

# Ensure target directories exist locally
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CSV = DATA_DIR / "karachi_featured_cache.csv"


# ─── AUTOMATED GITIGNORE SAFETY GUARD ────────────────────────────────────────
def enforce_gitignore():
    """Dynamically appends local storage folders to root .gitignore if missing."""
    gitignore_path = BASE_DIR / ".gitignore"
    rules_to_add = [
        "\n# EDA Engine Local Storage Guard",
        "eda/data/",
        "eda/outputs/",
        "*.csv",
        "*.json",
        "*.png"
    ]

    existing_content = ""
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            existing_content = f.read()

    # Filter rules that are not already present in .gitignore
    missing_rules = [rule for rule in rules_to_add if rule.strip() and rule.strip() not in existing_content]

    if missing_rules:
        with open(gitignore_path, "a") as f:
            if existing_content and not existing_content.endswith("\n"):
                f.write("\n")
            f.write("\n".join(rules_to_add) + "\n")
        print(f"🔒 Guard Active: Appended local data storage directories to {gitignore_path.name}")
    else:
        print("🔒 Guard Active: Workspace data directories are already secured in .gitignore")

# Automatically execute the safety check before pipeline data loads
enforce_gitignore()

# Sync with unified project-root environment keys
load_dotenv(BASE_DIR / ".env")
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME   = "karachi_aqi"
COLLECTION_NAME = "processed_features"

print("=" * 90)
print("             🚀 KARACHI AQI PREDICTOR — LOCAL STORAGE EDA WORKSPACE")
print("=" * 90)


# ─── 1. SECURE READ-ONLY DATA RETRIEVAL LAYER ─────────────────────────────────
def extract_feature_matrix() -> pd.DataFrame:
    """Queries the remote Atlas feature warehouse via a read-only request."""
    from pymongo import MongoClient
    if not MONGO_URI:
        raise ValueError(
            "CRITICAL: MONGODB_URI missing from environment setup.\n"
            "Please export your link or define it in your root .env file."
        )

    print("🛰️ Connecting to remote MongoDB Atlas Cluster (READ-ONLY)...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    db     = client[DB_NAME]
    col    = db[COLLECTION_NAME]

    total_records = col.count_documents({})
    print(f"  -> Found {total_records:,} engineered documents in '{DB_NAME}.{COLLECTION_NAME}'")

    if total_records == 0:
        raise ValueError("Target collection is unpopulated. Run 'run_feature_pipeline.py' first!")

    # Exclude internal MongoDB cursor IDs to keep dataframes clean
    docs = list(col.find({}, {"_id": 0}))
    client.close()
    print("  -> Download complete. Parsing memory structures...")
    return pd.DataFrame(docs)


# Implement local performance caching to guarantee instant execution on repeat runs
if LOCAL_CSV.exists():
    print(f"📦 Active Local Cache Detected. Loading data dynamically from:\n   {LOCAL_CSV}")
    df = pd.read_csv(LOCAL_CSV)
    print(f"   Parsed {len(df):,} matrix historical intervals across {len(df.columns)} columns.")
else:
    print("❌ No local tracking cache found. Commencing active cloud extraction...")
    df = extract_feature_matrix()
    df.to_csv(LOCAL_CSV, index=False)
    print(f"   ✓ Cache record generated successfully: {LOCAL_CSV} ({LOCAL_CSV.stat().st_size / 1e6:.1f} MB)")


# ─── 2. DTYPE NORMALISATION — coerce all numeric columns after CSV reload ─────
# CSV serialisation can silently downcast int/float columns to object dtype.
# Re-cast everything that looks numeric before any comparisons are made.
for col_name in df.columns:
    if df[col_name].dtype == object:
        converted = pd.to_numeric(df[col_name], errors="coerce")
        # Only replace if conversion is meaningful (>90 % non-NaN)
        if converted.notna().mean() > 0.9:
            df[col_name] = converted

print("🔧 Dtype Normalisation Pass Complete.")


# ─── 3. TEMPORAL INDEX INTEGRITY CHECK ───────────────────────────────────────
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    start_bound = df["datetime"].min().strftime("%Y-%m-%d %H:%M")
    end_bound   = df["datetime"].max().strftime("%Y-%m-%d %H:%M")
    print(f"⏰ Pipeline Timeline Windows Confirmed: {start_bound} ──► {end_bound}")
else:
    print("🚨 WARNING: Primary 'datetime' timestamp key missing. Time series charts skipped.")


# ─── 4. HIGH-LEVEL PIPELINE DATA MATRICES PROFILE (LOCAL LOGGING) ────────────
print("\n" + "─" * 60)
print(" ANALYTICAL PROFILE MATRIX SUMMARY (SAVING LOCALLY)")
print("─" * 60)

target_horizons   = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
available_targets = [t for t in target_horizons if t in df.columns]

metrics_profile = {
    "Total Observations (Hours)":   len(df),
    "Total Engineered Features":    len(df.columns),
    "Global Missing NaN Elements":  int(df.isna().sum().sum()),
    "Missing Structural Cell Ratio": float(df.isna().sum().sum() / df.size * 100),
    "Baseline AQI Mean":   float(df["aqi"].mean())   if "aqi" in df.columns else 0.0,
    "Baseline AQI Median": float(df["aqi"].median()) if "aqi" in df.columns else 0.0,
    "Baseline AQI Std Dev": float(df["aqi"].std())   if "aqi" in df.columns else 0.0,
}

for label, val in metrics_profile.items():
    fmt_str = (
        f"{val:,.2f}%" if "Ratio" in label
        else (f"{val:,.2f}" if isinstance(val, float) else f"{int(val):,}")
    )
    print(f"  ├── {label:<35}: {fmt_str}")

pd.DataFrame(list(metrics_profile.items()), columns=["Metric", "Value"]).to_csv(
    OUTPUT_DIR / "pipeline_profile_summary.csv", index=False
)
print(f"💾 Saved Dataset Profile summary to: {OUTPUT_DIR}/pipeline_profile_summary.csv")


# ─── 5. INVERSION PROXY ATMOSPHERIC HOVER AUDIT ───────────────────────────────
print("\n" + "─" * 60)
print(" INVERSION PROXY ATMOSPHERIC HOVER AUDIT")
print("─" * 60)

# ── Diagnostic dump so you can see actual value ranges before the filter runs ─
for diag_col in ["hour", "wind_speed_10m", "relative_humidity_2m"]:
    if diag_col in df.columns:
        col_s = pd.to_numeric(df[diag_col], errors="coerce")
        print(f"  [diag] {diag_col:<28} dtype={df[diag_col].dtype}  "
              f"min={col_s.min():.2f}  max={col_s.max():.2f}  "
              f"nulls={col_s.isna().sum()}")

inversion_cols = [c for c in df.columns if "inversion" in c or "proxy" in c]
proxy_feature  = inversion_cols[0] if inversion_cols else None

required_cols = {"hour", "wind_speed_10m", "aqi"}
if proxy_feature or required_cols.issubset(df.columns):

    if not proxy_feature:
        # FIX: Explicit numeric coercion prevents silent object-dtype comparison failures
        hour_col     = pd.to_numeric(df["hour"], errors="coerce")
        wind_col     = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
        humidity_col = pd.to_numeric(
            df["relative_humidity_2m"] if "relative_humidity_2m" in df.columns
            else pd.Series(0, index=df.index),
            errors="coerce"
        ).fillna(0)

        df["inversion_layer_proxy"] = (
            (hour_col.isin([5, 6, 7, 8, 9])) &
            (wind_col < 8.0) &
            (humidity_col > 70)
        ).astype(int)
        proxy_feature = "inversion_layer_proxy"

    proxy_active_pct   = (df[proxy_feature] == 1).mean() * 100
    mean_aqi_normal    = df[df[proxy_feature] == 0]["aqi"].mean()
    mean_aqi_inversion = df[df[proxy_feature] == 1]["aqi"].mean()

    print(f"  ├── Inversion Microclimate Active Windows : {proxy_active_pct:.2f}% of Total Timeline")

    # FIX: Guard NaN mean prints so output is always readable
    normal_str    = f"{mean_aqi_normal:.2f}"    if pd.notna(mean_aqi_normal)    else "N/A (no samples)"
    inversion_str = f"{mean_aqi_inversion:.2f}" if pd.notna(mean_aqi_inversion) else "N/A (no samples)"
    print(f"  ├── Mean AQI during Non-Inversion periods  : {normal_str}")
    print(f"  └── Mean AQI during Active Trapping Inversion: {inversion_str}")

    # FIX: Only write JSON and plot when both groups exist and have valid means
    unique_proxy_vals = sorted(df[proxy_feature].dropna().unique().tolist())
    both_groups_exist = (0 in unique_proxy_vals) and (1 in unique_proxy_vals)
    both_means_valid  = pd.notna(mean_aqi_normal) and pd.notna(mean_aqi_inversion)

    if not both_groups_exist:
        print(f"  ⚠️  Skipping inversion outputs — only class(es) {unique_proxy_vals} present in proxy column.")
        print("      Possible causes: 'hour' values outside [5-9], wind_speed ≥ 8.0 always, "
              "or humidity ≤ 70 always. Review diagnostic lines above.")
    else:
        if both_means_valid:
            inversion_report = {
                "inversion_active_percentage":  round(proxy_active_pct, 2),
                "mean_aqi_normal_dispersion":   round(mean_aqi_normal, 2),
                "mean_aqi_trapped_inversion":   round(mean_aqi_inversion, 2),
                "calculated_aqi_lift_factor":   round(mean_aqi_inversion / max(1, mean_aqi_normal), 2),
            }
            with open(OUTPUT_DIR / "inversion_metrics_summary.json", "w") as f:
                json.dump(inversion_report, f, indent=4)
            print(f"  ✓ Inversion metrics JSON saved.")

        # FIX: Build palette only from classes actually present — prevents seaborn KeyError
        full_palette   = {0: "#2b7bba", 1: "#e05a47"}
        safe_palette   = {k: v for k, v in full_palette.items() if k in unique_proxy_vals}
        label_map      = {0: "Standard Dispersion", 1: "Active Surface Inversion"}
        x_tick_labels  = [label_map[k] for k in unique_proxy_vals]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            x=proxy_feature, y="aqi", data=df, ax=ax,
            palette=safe_palette, order=unique_proxy_vals
        )
        ax.set_xticks(range(len(unique_proxy_vals)))
        ax.set_xticklabels(x_tick_labels)
        ax.set_title(
            "Karachi Atmospheric Mixing Layer Proxy Impact\n"
            "(Shallow Mixing Traps Surface Particulates)",
            fontweight="bold"
        )
        ax.set_ylabel("Measured AQI Levels")
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "inversion_proxy_impact.png", dpi=300)
        plt.close()
        print(f"  ✓ Inversion boxplot saved.")

else:
    print("  ⚠️  Required columns missing for inversion proxy. Skipping section.")


# ─── 6. MULTI-HORIZON CRITICAL TARGET DRIFT EVALUATION ────────────────────────
if len(available_targets) > 0 and "datetime" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    melted_targets = df.melt(
        id_vars=["datetime"],
        value_vars=available_targets,
        var_name="Horizon",
        value_name="Target_AQI"
    )
    sns.kdeplot(
        data=melted_targets, x="Target_AQI", hue="Horizon",
        common_norm=False, fill=True, alpha=0.2, linewidth=2,
        palette="viridis", ax=ax
    )
    ax.set_title("Predictive Target Density Distribution Across Forecast Horizons", fontweight="bold")
    ax.set_xlabel("Target AQI Vector Values")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_horizon_target_density.png", dpi=300)
    plt.close()
    print("✓ Multi-horizon target density chart saved.")


# ─── 7. MONSOON AND WINTER REGIME FOURIER SEASONALITY PROFILE ─────────────────
fourier_cols = [c for c in df.columns if "sin" in c or "cos" in c]
if fourier_cols and "datetime" in df.columns:
    seasonal_df = df.set_index("datetime").resample("D")[["aqi"]].mean().reset_index()
    seasonal_df["Month_Name"] = seasonal_df["datetime"].dt.strftime("%b")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x="Month_Name", y="aqi", data=seasonal_df, ax=ax,
        palette="coolwarm",
        order=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_title(
        "Karachi Macro Environmental Seasonality Trends\n"
        "(High-Density Winter Haze vs Summer Monsoon Cleansing Regimes)",
        fontweight="bold"
    )
    ax.set_ylabel("Daily Average Base AQI")
    ax.set_xlabel("Annual Phase Execution Month")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "karachi_macro_seasonality_profile.png", dpi=300)
    plt.close()
    print("✓ Macro seasonality profile chart saved.")


# ─── 8. MULTI-HORIZON FORECAST CORES CORRELATION TARGET MATRIX ────────────────
all_possible_stales = [
    "target_aqi_12h_log", "target_aqi_24h_log", "target_aqi_48h_log", "target_aqi_72h_log",
    "target_cat_24h", "target_cat_48h", "target_cat_72h"
]
numeric_clean = df.select_dtypes(include=[np.number]).drop(
    columns=[col for col in all_possible_stales if col in df.columns],
    errors="ignore"
)

if available_targets and "aqi" in numeric_clean.columns:
    print("\n" + "─" * 60)
    print(" TARGET VECTOR COEFFICIENT EXTRACTS (LOCAL CSVs EXPORTED)")
    print("─" * 60)

    corr_matrix = numeric_clean.corr()

    for target in available_targets:
        if target not in corr_matrix.columns:
            print(f"  ⚠️  [{target}] not found in numeric correlation matrix. Skipping.")
            continue

        print(f"📊 Top Drivers correlated against [{target}]:")
        top_drivers = (
            corr_matrix[target]
            .drop(index=[t for t in target_horizons + ["aqi"] if t in corr_matrix.index], errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(8)
        )
        signed_drivers = corr_matrix[target].loc[top_drivers.index]

        for feat, score in signed_drivers.items():
            print(f"  ├── {feat:<35}: {score:+.4f}")

        signed_drivers.to_csv(OUTPUT_DIR / f"drivers_summary_{target}.csv")
        print("  " + "┈" * 45)

    core_vis_features = [
        "aqi", "pm25", "pm10", "temperature_2m",
        "relative_humidity_2m", "wind_speed_10m"
    ] + available_targets
    vis_intersect = [f for f in core_vis_features if f in numeric_clean.columns]

    if len(vis_intersect) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            numeric_clean[vis_intersect].corr(),
            annot=True, cmap="mako_r", fmt=".2f",
            square=True, cbar_kws={"shrink": .75}, ax=ax
        )
        ax.set_title(
            "Karachi Pipeline Features & Unified Forecast Horizon Core Correlation Map",
            fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "horizon_cross_correlation_matrix.png", dpi=300)
        plt.close()
        print("✓ Horizon cross-correlation heatmap saved.")


# ─── 9. GENERATE TRANSITIONAL STRUCTURE STATISTICAL MATRICES ──────────────────
numeric_clean.describe().T.to_csv(OUTPUT_DIR / "pipeline_feature_descriptives.csv")
print("✓ Full feature descriptive statistics exported.")

print("\n" + "=" * 90)
print(f"✅ EXHAUSTIVE WORKSPACE EDA PIPELINE RUN COMPLETE.")
print(f"   Storage Guard : 0 database insertions executed.")
print(f"   Outputs safely written into your local directory: {OUTPUT_DIR.resolve()}")
print("=" * 90)
