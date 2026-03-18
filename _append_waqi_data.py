"""
_append_waqi_data.py — Fetch latest WAQI readings and append to featured.csv.

Run from the odisha-aqi-advisor/ directory:
    python _append_waqi_data.py

Steps:
1. Fetch current AQI for all 10 cities via WAQI API
2. Compute lag/rolling features from the tail of existing featured.csv
3. Skip rows already present (dedup by city + date)
4. Append new rows to featured.csv
5. Write a fetch log to data/processed/waqi_fetch_log.txt
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.waqi_fetcher import fetch_all_cities, WAQI_CITY_NAMES
from src.data_loader import CITIES, INDUSTRIAL_CITIES, DIWALI_DATES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FEATURED_PATH = PROJECT_ROOT / "data" / "processed" / "featured.csv"
LOG_PATH      = PROJECT_ROOT / "data" / "processed" / "waqi_fetch_log.txt"

PEAK_MONTHS = {10, 11, 12, 1, 2}
DIWALI_TS   = [pd.Timestamp(d) for d in DIWALI_DATES]


def _is_diwali_week(dt: pd.Timestamp) -> int:
    return int(any(abs((dt - d).days) <= 3 for d in DIWALI_TS))


def compute_features(new_rows: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Compute all feature columns for new_rows using existing data as history."""
    result_rows = []

    for _, row in new_rows.iterrows():
        city = row["city"]
        dt   = row["date"]

        # History for this city (sorted)
        hist = existing[existing["city"] == city].sort_values("date")

        # Lag-1 features: use last known row
        if len(hist) > 0:
            last = hist.iloc[-1]
            aqi_yesterday = last["aqi"]
            pm25_lag1     = last["pm25"]
            pm10_lag1     = last["pm10"]
            so2_lag1      = last["so2"]
            no2_lag1      = last["no2"]
            # 7-day rolling avg of aqi (last 7 rows)
            aqi_7day_avg  = hist["aqi"].tail(7).mean()
        else:
            aqi_yesterday = np.nan
            pm25_lag1 = pm10_lag1 = so2_lag1 = no2_lag1 = np.nan
            aqi_7day_avg = np.nan

        month               = dt.month
        is_winter           = int(month in {11, 12, 1})
        is_monsoon          = int(month in {7, 8, 9})
        is_industrial_peak  = int(city in INDUSTRIAL_CITIES and month in PEAK_MONTHS)

        # aqi_target: we don't know tomorrow's AQI yet — leave as NaN
        result_rows.append({
            **row.to_dict(),
            "data_quality":        row.get("data_quality", "waqi_live"),
            "aqi_yesterday":       round(aqi_yesterday, 2) if not np.isnan(aqi_yesterday) else np.nan,
            "pm25_lag1":           round(pm25_lag1, 2)     if pm25_lag1 is not None and not np.isnan(pm25_lag1) else np.nan,
            "pm10_lag1":           round(pm10_lag1, 2)     if pm10_lag1 is not None and not np.isnan(pm10_lag1) else np.nan,
            "so2_lag1":            round(so2_lag1, 2)      if so2_lag1  is not None and not np.isnan(so2_lag1)  else np.nan,
            "no2_lag1":            round(no2_lag1, 2)      if no2_lag1  is not None and not np.isnan(no2_lag1)  else np.nan,
            "aqi_7day_avg":        round(aqi_7day_avg, 2)  if not np.isnan(aqi_7day_avg) else np.nan,
            "aqi_target":          np.nan,
            "month":               month,
            "is_winter":           is_winter,
            "is_monsoon":          is_monsoon,
            "is_industrial_peak":  is_industrial_peak,
        })

    return pd.DataFrame(result_rows)


def main():
    log_lines = [f"=== WAQI fetch run: {datetime.now().isoformat()} ==="]

    # Load existing featured.csv
    if not FEATURED_PATH.exists():
        log.error("featured.csv not found at %s", FEATURED_PATH)
        sys.exit(1)

    existing = pd.read_csv(FEATURED_PATH, parse_dates=["date"])
    log.info("Existing featured.csv: %d rows, max date: %s", len(existing), existing["date"].max().date())
    log_lines.append(f"Existing rows: {len(existing)}, max date: {existing['date'].max().date()}")

    # Fetch new data
    log.info("Fetching WAQI data for all cities...")
    new_raw = fetch_all_cities(delay=0.5)

    if new_raw.empty:
        msg = "No data fetched from WAQI — all cities failed."
        log.warning(msg)
        log_lines.append(msg)
        LOG_PATH.write_text("\n".join(log_lines))
        sys.exit(0)

    log_lines.append(f"WAQI rows fetched: {len(new_raw)}")
    for _, r in new_raw.iterrows():
        log_lines.append(f"  {r['city']}: AQI={r['aqi']} on {r['date'].date()}")

    # Dedup: skip rows already in existing (same city + date)
    existing_keys = set(zip(existing["city"], existing["date"].dt.date.astype(str)))
    new_raw["_date_str"] = new_raw["date"].dt.date.astype(str)
    new_raw = new_raw[~new_raw.apply(lambda r: (r["city"], r["_date_str"]) in existing_keys, axis=1)]
    new_raw = new_raw.drop(columns=["_date_str"])

    if new_raw.empty:
        msg = "All fetched rows already exist in featured.csv — nothing to append."
        log.info(msg)
        log_lines.append(msg)
        LOG_PATH.write_text("\n".join(log_lines))
        print(f"\nMax date in featured.csv: {existing['date'].max().date()}")
        sys.exit(0)

    log.info("%d new rows to append", len(new_raw))
    log_lines.append(f"New rows after dedup: {len(new_raw)}")

    # Compute features
    featured_new = compute_features(new_raw, existing)

    # Ensure column order matches existing
    col_order = existing.columns.tolist()
    for col in col_order:
        if col not in featured_new.columns:
            featured_new[col] = np.nan
    featured_new = featured_new[col_order]

    # Append and save
    combined = pd.concat([existing, featured_new], ignore_index=True)
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)
    combined.to_csv(FEATURED_PATH, index=False)

    new_max = combined["date"].max().date()
    msg = f"Appended {len(featured_new)} rows. New max date: {new_max}"
    log.info(msg)
    log_lines.append(msg)

    LOG_PATH.write_text("\n".join(log_lines))
    print(f"\nMax date in featured.csv: {new_max}")


if __name__ == "__main__":
    main()
