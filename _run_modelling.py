"""Standalone script: train all models, save artefacts (V3 — Plotly, no PNG charts)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import joblib as jl
from src.features import FEATURE_COLS
from src.model import (
    chronological_split, train_linear, train_xgboost,
    evaluate_model, save_model, load_and_verify,
)

FEATURED_CSV = "data/processed/featured.csv"
MODELS_DIR   = "models"
RESULTS_CSV  = "data/processed/model_results.csv"
FEAT_JSON    = "models/feature_columns.json"
TARGET_COL   = "aqi_target"

os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(FEATURED_CSV, parse_dates=["date"])
cities = sorted(df["city"].unique())
print(f"Cities ({len(cities)}): {cities}")

results = []
xgb_models = {}

for city in cities:
    city_df = (
        df[df["city"] == city]
        .sort_values("date")
        .reset_index(drop=True)
        .dropna(subset=FEATURE_COLS + [TARGET_COL])
    )
    train_df, test_df = chronological_split(city_df, test_fraction=0.2)
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].values

    # Linear Regression
    lr = train_linear(X_train, y_train)
    lr_m = evaluate_model(lr, X_test, y_test)
    save_model(lr, os.path.join(MODELS_DIR, f"lr_{city.lower()}.joblib"))
    results.append({"city": city, "model_type": "lr", **lr_m})

    # XGBoost
    try:
        xgb = train_xgboost(X_train, y_train)
        xgb_m = evaluate_model(xgb, X_test, y_test)
        save_model(xgb, os.path.join(MODELS_DIR, f"xgb_{city.lower()}.joblib"))
        results.append({"city": city, "model_type": "xgb", **xgb_m})
        xgb_models[city] = xgb
        print(f"  {city}: LR MAE={lr_m['mae']:.2f}  XGB MAE={xgb_m['mae']:.2f}")
    except Exception as e:
        print(f"  WARNING XGBoost {city}: {e}")
        print(f"  {city}: LR MAE={lr_m['mae']:.2f}")

# Write feature_columns.json
with open(FEAT_JSON, "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"\nWritten {FEAT_JSON} ({len(FEATURE_COLS)} cols)")

# Write model_results.csv
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"Written {RESULTS_CSV} ({len(results_df)} rows)")
print(results_df.to_string(index=False))

# Round-trip verification
all_ok = True
for city in cities:
    city_df = (
        df[df["city"] == city]
        .sort_values("date")
        .reset_index(drop=True)
        .dropna(subset=FEATURE_COLS + [TARGET_COL])
    )
    _, test_df = chronological_split(city_df, test_fraction=0.2)
    X_test = test_df[FEATURE_COLS].values
    for mt in ["lr", "xgb"]:
        path = os.path.join(MODELS_DIR, f"{mt}_{city.lower()}.joblib")
        if os.path.exists(path):
            m = jl.load(path)
            if not load_and_verify(m, path, X_test):
                print(f"MISMATCH: {path}")
                all_ok = False

print(f"\nRound-trip verification: {'PASSED' if all_ok else 'FAILED'}")
print(f".joblib count: {len(glob.glob(os.path.join(MODELS_DIR, '*.joblib')))}")
