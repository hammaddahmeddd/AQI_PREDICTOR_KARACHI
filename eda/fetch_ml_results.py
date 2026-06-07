"""
eda/fetch_ml_results.py
──────────────────────────────────────────────────────────────────────────────
Extracts model telemetry, cross-model metrics, and SHAP configurations directly
from your MongoDB backend and materializes them into your local eda/outputs/ folder.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Setup folder directories
EDA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EDA_DIR.parent
OUTPUT_DIR = EDA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load database environment variables
load_dotenv(PROJECT_ROOT / ".env")
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "karachi_aqi"

def get_mongo_client():
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGODB_URI is empty or missing from your environment config.")
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

def extract_and_visualize():
    print("=" * 80)
    print(" 🛰️  MONGO PIPELINE TELEMETRY & SHAP ANALYTICS EXTRACTOR")
    print("=" * 80)
    
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        
        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: FETCH CROSS-MODEL SUMMARY & EVALUATION DATA
        # ──────────────────────────────────────────────────────────────────────
        print("\n[Step 1] Extracting cross-model validation logs...")
        
        # Pull the latest run tracking document containing performance metrics
        pipeline_doc = db["pipeline_status"].find_one(
            {"horizons": {"$exists": True}},
            sort=[("logged_at", DESCENDING)]
        )
        
        if not pipeline_doc:
            print("  ⚠️ No structured training history summaries found in 'pipeline_status'.")
            print("  💡 Tip: Run your training pipeline locally once to generate logs: python run_training_pipeline.py")
            perf_data = None
        else:
            print(f"  -> Successfully located training matrix timestamped: {pipeline_doc.get('logged_at')}")
            perf_data = pipeline_doc.get("horizons", {})

        if perf_data:
            rows = []
            for horizon_key, horizon_data in perf_data.items():
                horizon_num = horizon_key.replace("h", "")
                models_dict = horizon_data.get("models", {})
                best_model = horizon_data.get("best_model", "N/A")
                
                for model_name, metrics in models_dict.items():
                    rows.append({
                        "Horizon": f"{horizon_num} Hours",
                        "ML_Model": model_name.upper().replace("_", " "),
                        "Is_Winner": (model_name == best_model),
                        "R2_Score": metrics.get("r2") or metrics.get("R2") or 0.0,
                        "RMSE": metrics.get("rmse") or metrics.get("RMSE") or 0.0
                    })
            
            if rows:
                df_perf = pd.DataFrame(rows)
                df_perf["R2_Score"] = pd.to_numeric(df_perf["R2_Score"], errors='coerce').fillna(0.0)
                
                # Save data matrix to CSV
                csv_path = OUTPUT_DIR / "pipeline_model_comparison.csv"
                df_perf.to_csv(csv_path, index=False)
                print(f"  ✓ Saved cross-model performance ledger: {csv_path}")
                
                # Generate Performance Comparison Chart
                plt.figure(figsize=(10, 5))
                sns.set_theme(style="whitegrid")
                sns.barplot(
                    x="Horizon", 
                    y="R2_Score", 
                    hue="ML_Model", 
                    data=df_perf, 
                    palette="coolwarm"
                )
                plt.title("Karachi AQI Predictive Capability ($R^2$ Variance Analysis)", fontsize=12, fontweight="bold")
                plt.ylabel("Testing Validation Coefficient ($R^2$)")
                plt.xlabel("Forecasting Horizon Windows")
                plt.ylim(0, 1.0)
                plt.legend(title="Model Frameworks", loc="lower left")
                plt.tight_layout()
                
                chart_path = OUTPUT_DIR / "model_r2_comparison_matrix.png"
                plt.savefig(chart_path, dpi=300)
                plt.close()
                print(f"  ✓ Rendered visualization asset: {chart_path.name}")

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: FETCH GLOBAL SHAP IMPORTANCES 
        # ──────────────────────────────────────────────────────────────────────
        print("\n[Step 2] Processing SHAP feature attribution matrices...")
        
        # Fallback tracking if a dedicated top_features table isn't populated yet
        all_shap_rows = []
        
        # Attempt to scan standard collection targets
        for col_name in ["top_features", "feature_importance"]:
            if db[col_name].count_documents({}) > 0:
                print(f"  -> Extracting logs directly from collection: '{col_name}'")
                for doc in db[col_name].find({}):
                    model = doc.get("model", "Unknown")
                    horizon = doc.get("horizon", "Unknown")
                    features = doc.get("features", doc.get("top_features", []))
                    for f in features:
                        all_shap_rows.append({
                            "Model": model.upper(),
                            "Horizon": f"{horizon}h" if "h" not in str(horizon) else str(horizon),
                            "Feature": f.get("feature") or f.get("Feature"),
                            "SHAP_Value": float(f.get("mean_abs_shap") or f.get("importance") or 0)
                        })
                break
        
        # If empty, parse feature lists out of the recent pipeline documentation
        if not all_shap_rows:
            print("  -> Scanning nested historical pipelines for feature rankings...")
            recent_runs = db["pipeline_status"].find({"features": {"$exists": True}}).sort("logged_at", DESCENDING).limit(5)
            for doc in recent_runs:
                model = doc.get("model", "Model Portfolio")
                horizon = doc.get("horizon", "24")
                for f in doc.get("features", []):
                    all_shap_rows.append({
                        "Model": str(model).upper(),
                        "Horizon": f"{horizon}h",
                        "Feature": f.get("feature"),
                        "SHAP_Value": float(f.get("mean_abs_shap") or 0)
                    })

        if all_shap_rows:
            df_shap = pd.DataFrame(all_shap_rows)
            
            # Save master metadata file
            shap_csv = OUTPUT_DIR / "global_feature_importance_registry.csv"
            df_shap.to_csv(shap_csv, index=False)
            print(f"  ✓ Saved SHAP importance lookup table: {shap_csv}")
            
            # Draw individual horizontal bar charts for each discovered forecast slice
            for horizon_lbl in df_shap["Horizon"].unique():
                slice_df = df_shap[df_shap["Horizon"] == horizon_lbl].copy()
                slice_df = slice_df.sort_values(by="SHAP_Value", ascending=False).head(12)
                
                if not slice_df.empty:
                    plt.figure(figsize=(9, 5))
                    sns.barplot(
                        x="SHAP_Value",
                        y="Feature",
                        data=slice_df,
                        palette="viridis"
                    )
                    plt.title(f"Explainable AI Profile: Top 12 Global SHAP Drivers ({horizon_lbl})", fontsize=11, fontweight="bold")
                    plt.xlabel("Mean Absolute Impact on Air Quality Index Prediction (Points)")
                    plt.ylabel("")
                    plt.tight_layout()
                    
                    output_img = OUTPUT_DIR / f"shap_drivers_profile_{horizon_lbl}.png"
                    plt.savefig(output_img, dpi=300)
                    plt.close()
                    print(f"  ✓ Rendered diagnostic chart asset: {output_img.name}")
        else:
            print("  ⚠️ No global SHAP attribution datasets discovered in MongoDB records.")
            print("  💡 Tip: Your training metrics code hasn't pushed records since SHAP tracking was enabled.")

        print("\n========================================================================")
        print("🎉 WORKFLOW COMPLETE: Local EDA outputs folder successfully updated.")
        print("========================================================================")

    except PyMongoError as e:
        print(f"\n❌ Remote MongoDB Atlas Extraction Error: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    extract_and_visualize()