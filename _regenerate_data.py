"""
_regenerate_data.py — Regenerate unified_clean.csv and featured.csv up to today.

Run from the odisha-aqi-advisor/ directory:
    python _regenerate_data.py

This replaces notebooks 01 + 02 for the purpose of extending the date range.
Uses the same synthetic data logic as notebook 01 but with:
- Current CITIES dict (V3 city names)
- DATE_END = today
"""
import sys
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CITIES, INDUSTRIAL_CITIES, DIWALI_DATES
from src.features import run_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATE_START = "2019-01-01"
DATE_END   = date.today().isoformat()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_PATH    = PROCESSED_DIR / "unified_clean.csv"
FEATURED_PATH = PROCESSED_DIR / "featured.csv"

# City-level synthetic data config (matches V3 city names)
CITY_CONFIG = {
    "Jharsuguda":  {"base": 180, "std": 60, "tier": "industrial"},
    "Angul":       {"base": 170, "std": 55, "tier": "industrial"},
    "Talcher":     {"base": 190, "std": 65, "tier": "industrial"},
    "Rourkela":    {"base": 160, "std": 50, "tier": "industrial"},
    "Sambalpur":   {"base": 150, "std": 50, "tier": "industrial"},
    "Bhubaneswar": {"base": 100, "std": 35, "tier": "urban"},
    "Cuttack":     {"base": 110, "std": 38, "tier": "urban"},
    "Balasore":    {"base": 95,  "std": 32, "tier": "urban"},
    "Berhampur":   {"base": 65,  "std": 22, "tier": "clean"},
    "Rayagada":    {"base": 55,  "std": 18, "tier": "clean"},
}

DIWALI_TS = pd.to_datetime(DIWALI_DATES)


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(DATE_START, DATE_END, freq="D")
    rows  = []

    for city, cfg in CITY_CONFIG.items():
        base = cfg["base"]
        std  = cfg["std"]

        for d in dates:
            month = d.month

            # Seasonal multiplier
            if month in (11, 12, 1, 2):
                seasonal = 1.30
            elif month in (7, 8, 9):
                seasonal = 0.70
            elif month in (3, 4, 5):
                seasonal = 1.10
            else:
                seasonal = 1.00

            # Diwali spike
            diwali_spike = 1.0
            for dw in DIWALI_TS:
                if abs((d - dw).days) <= 3:
                    diwali_spike = 1.40
                    break

            aqi_mean = base * seasonal * diwali_spike
            aqi = float(rng.normal(aqi_mean, std))

            if cfg["tier"] == "industrial":
                aqi = np.clip(aqi, 80, 350)
            elif cfg["tier"] == "urban":
                aqi = np.clip(aqi, 50, 200)
            else:
                aqi = np.clip(aqi, 30, 120)

            pm25 = float(np.clip(rng.normal(aqi * 0.45, aqi * 0.05), 0, 500))
            pm10 = float(np.clip(rng.normal(aqi * 0.70, aqi * 0.07), 0, 600))
            no2  = float(np.clip(rng.normal(aqi * 0.18, aqi * 0.03), 0, 400))
            so2  = float(np.clip(rng.normal(aqi * 0.12, aqi * 0.02), 0, 800))
            o3   = float(np.clip(rng.normal(aqi * 0.10, aqi * 0.02), 0, 400))
            co   = float(np.clip(rng.normal(aqi * 0.008, aqi * 0.001), 0, 50))

            rows.append({
                "city": city, "date": d,
                "aqi":  round(aqi, 2),
                "pm25": round(pm25, 2), "pm10": round(pm10, 2),
                "no2":  round(no2, 2),  "so2":  round(so2, 2),
                "o3":   round(o3, 2),   "co":   round(co, 2),
            })

    df = pd.DataFrame(rows)
    df["data_quality"] = "original"
    log.info("Synthetic data: %d rows, %d cities, %s → %s",
             len(df), df["city"].nunique(), DATE_START, DATE_END)
    return df


def main():
    log.info("Generating synthetic data from %s to %s ...", DATE_START, DATE_END)
    clean = generate_synthetic_data()
    clean = clean.sort_values(["city", "date"]).reset_index(drop=True)
    clean.to_csv(CLEAN_PATH, index=False)
    log.info("Saved unified_clean.csv: %d rows", len(clean))

    log.info("Running feature engineering pipeline ...")
    featured = run_full_pipeline(clean)
    featured.to_csv(FEATURED_PATH, index=False)
    log.info("Saved featured.csv: %d rows", len(featured))

    max_date = featured["date"].max().date()
    print(f"\nMax date in featured.csv: {max_date}")
    print(f"Total rows: {len(featured)}")


if __name__ == "__main__":
    main()
