"""
data_collector.py — Dual data pipeline: CPCB CSV + OpenAQ API.
Merges both sources into data/raw/unified_raw.csv.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.data_loader import CITIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CPCB AQI breakpoints: (pollutant, [(conc_lo, conc_hi, idx_lo, idx_hi), ...])
# ---------------------------------------------------------------------------
_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
        (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500),
    ],
    "pm10": [
        (0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
        (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500),
    ],
    "so2": [
        (0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200),
        (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2100, 400, 500),
    ],
    "no2": [
        (0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200),
        (180, 280, 200, 300), (280, 400, 300, 400), (400, 600, 400, 500),
    ],
}

# OpenAQ v3 location IDs for Odisha cities (best-effort; None = no coverage)
_OPENAQ_LOCATION_IDS = {
    "Bhubaneswar": 8118,
    "Cuttack":     None,
    "Balasore":    None,
    "Jharsuguda":  None,
    "Angul":       None,
    "Talcher":     None,
    "Rourkela":    None,
    "Sambalpur":   None,
    "Berhampur":   None,
    "Rayagada":    None,
}


def _sub_index(value: float, breakpoints: list) -> float:
    """Linear interpolation of sub-index from CPCB breakpoint table."""
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= value <= c_hi:
            return i_lo + (value - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    # Above highest breakpoint
    return 500.0


def calculate_aqi(
    pm25: Optional[float],
    pm10: Optional[float],
    so2: Optional[float],
    no2: Optional[float],
) -> Optional[float]:
    """Compute AQI from pollutant concentrations using CPCB sub-index formula.

    Returns the maximum sub-index across all non-null pollutants.
    Returns None if all inputs are None or NaN.
    """
    sub_indices = []
    for val, key in [(pm25, "pm25"), (pm10, "pm10"), (so2, "so2"), (no2, "no2")]:
        if val is not None and not (isinstance(val, float) and np.isnan(val)) and val >= 0:
            sub_indices.append(_sub_index(float(val), _BREAKPOINTS[key]))
    if not sub_indices:
        return None
    return max(sub_indices)


def load_cpcb_csv(city: str, raw_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """Load data/raw/{city_lower}_cpcb.csv. Returns None and logs warning if missing."""
    path = os.path.join(raw_dir, f"{city.lower()}_cpcb.csv")
    if not os.path.exists(path):
        logger.warning("CPCB CSV not found for %s at %s", city, path)
        return None
    df = pd.read_csv(path)
    required = {"date", "city", "pm25", "pm10", "so2", "no2", "aqi"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("CPCB CSV for %s missing columns: %s", city, missing)
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["source"] = "cpcb"
    return df


def fetch_openaq_data(city: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Fetch last `days` days of pollutant readings from OpenAQ v3 API.

    Returns DataFrame with columns [date, pm25, pm10, so2, no2] or None on failure.
    """
    location_id = _OPENAQ_LOCATION_IDS.get(city)
    if location_id is None:
        return None

    date_to = datetime.utcnow()
    date_from = date_to - timedelta(days=days)
    url = f"https://api.openaq.org/v3/locations/{location_id}/measurements"
    params = {
        "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit":     10000,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("OpenAQ timeout for %s", city)
        return None
    except requests.exceptions.HTTPError as e:
        logger.error("OpenAQ %s for %s", e.response.status_code, city)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("OpenAQ request error for %s: %s", city, e)
        return None

    data = resp.json().get("results", [])
    if not data:
        return None

    records = []
    for m in data:
        records.append({
            "date":      m.get("date", {}).get("utc", "")[:10],
            "parameter": m.get("parameter", ""),
            "value":     m.get("value"),
        })
    raw = pd.DataFrame(records)
    if raw.empty:
        return None

    # Pivot to wide format
    pivot = raw.pivot_table(index="date", columns="parameter", values="value", aggfunc="mean")
    pivot = pivot.reset_index()
    for col in ["pm25", "pm10", "so2", "no2"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["city"] = city
    pivot["source"] = "openaq"
    return pivot[["date", "city", "pm25", "pm10", "so2", "no2"]]


def merge_sources(
    cities: Optional[list] = None,
    raw_dir: str = "data/raw",
    log_path: str = "data/data_sources_log.txt",
) -> pd.DataFrame:
    """Merge CPCB CSV + OpenAQ data for all cities, save unified_raw.csv.

    CPCB values take precedence on duplicate (city, date) rows.
    """
    if cities is None:
        cities = list(CITIES.keys())

    all_frames = []
    log_lines = []

    for city in cities:
        cpcb_df = load_cpcb_csv(city, raw_dir)
        oaq_df  = fetch_openaq_data(city)

        if cpcb_df is not None and oaq_df is not None:
            # Combine; CPCB preferred on conflict
            combined = pd.concat([cpcb_df, oaq_df], ignore_index=True)
            combined = combined.sort_values("source")  # "cpcb" < "openaq"
            combined = combined.drop_duplicates(subset=["city", "date"], keep="first")
            log_lines.append(f"{city}: CPCB + OpenAQ")
        elif cpcb_df is not None:
            combined = cpcb_df
            log_lines.append(f"{city}: CPCB only")
        elif oaq_df is not None:
            oaq_df["aqi"] = oaq_df.apply(
                lambda r: calculate_aqi(r.get("pm25"), r.get("pm10"), r.get("so2"), r.get("no2")),
                axis=1,
            )
            combined = oaq_df
            log_lines.append(f"{city}: OpenAQ only")
        else:
            log_lines.append(f"{city}: no data")
            continue

        all_frames.append(combined)

    if not all_frames:
        logger.warning("No data collected for any city.")
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["city", "date"]).reset_index(drop=True)

    out_path = os.path.join(raw_dir, "unified_raw.csv")
    merged.to_csv(out_path, index=False)
    logger.info("Saved unified_raw.csv with %d rows", len(merged))

    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return merged
