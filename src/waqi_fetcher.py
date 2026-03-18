"""
waqi_fetcher.py — Fetch historical AQI data from the WAQI API.

Uses the World Air Quality Index (WAQI) feed API to pull daily AQI readings
for Odisha cities and return them in the project's standard schema.

Token: stored in WAQI_TOKEN constant (or override via env var WAQI_TOKEN).
"""
import os
import time
import logging
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "ef5feda541fb94c36eb71257d7687ef36bc929ca")

# Mapping from project city name → WAQI station slug(s) to try in order
WAQI_CITY_NAMES: dict[str, list[str]] = {
    "Jharsuguda":  ["jharsuguda", "india/jharsuguda"],
    "Angul":       ["angul", "india/angul"],
    "Talcher":     ["talcher", "india/talcher"],
    "Rourkela":    ["rourkela", "india/rourkela"],
    "Sambalpur":   ["sambalpur", "india/sambalpur"],
    "Bhubaneswar": ["bhubaneswar", "india/bhubaneswar"],
    "Cuttack":     ["cuttack", "india/cuttack"],
    "Balasore":    ["balasore", "india/balasore"],
    "Berhampur":   ["brahmapur", "india/brahmapur", "berhampur", "india/berhampur"],
    "Rayagada":    ["rayagada", "india/rayagada"],
}

BASE_URL = "https://api.waqi.info/feed"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_station(slug: str) -> Optional[dict]:
    """Fetch current feed for a station slug. Returns parsed JSON data or None."""
    url = f"{BASE_URL}/{slug}/"
    try:
        resp = requests.get(url, params={"token": WAQI_TOKEN}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "ok":
            return payload["data"]
    except Exception as exc:
        log.debug("WAQI fetch failed for %s: %s", slug, exc)
    return None


def _extract_iaqi(data: dict, key: str) -> Optional[float]:
    """Safely extract a pollutant value from the iaqi dict."""
    try:
        return float(data["iaqi"][key]["v"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_waqi_city(city_name: str) -> Optional[dict]:
    """Fetch the latest AQI reading for a city.

    Tries each slug in WAQI_CITY_NAMES[city_name] until one succeeds.

    Returns a dict with keys: city, date, aqi, pm25, pm10, no2, so2, o3, co
    or None if all slugs fail.
    """
    slugs = WAQI_CITY_NAMES.get(city_name, [city_name.lower()])
    for slug in slugs:
        data = _fetch_station(slug)
        if data is None:
            continue
        try:
            aqi_val = float(data["aqi"])
        except (KeyError, TypeError, ValueError):
            continue

        # Date from API (format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS+HH:MM")
        raw_time = data.get("time", {}).get("s", "")
        try:
            obs_date = pd.Timestamp(raw_time).date()
        except Exception:
            obs_date = date.today()

        row = {
            "city":  city_name,
            "date":  pd.Timestamp(obs_date),
            "aqi":   round(aqi_val, 2),
            "pm25":  _extract_iaqi(data, "pm25"),
            "pm10":  _extract_iaqi(data, "pm10"),
            "no2":   _extract_iaqi(data, "no2"),
            "so2":   _extract_iaqi(data, "so2"),
            "o3":    _extract_iaqi(data, "o3"),
            "co":    _extract_iaqi(data, "co"),
        }
        log.info("Fetched %s → AQI %.0f on %s (slug: %s)", city_name, aqi_val, obs_date, slug)
        return row

    log.warning("All slugs failed for city: %s", city_name)
    return None


def fetch_all_cities(delay: float = 0.5) -> pd.DataFrame:
    """Fetch latest AQI for all 10 cities. Returns a DataFrame.

    Parameters
    ----------
    delay : float
        Seconds to wait between API calls to avoid rate limiting.
    """
    rows = []
    for city in WAQI_CITY_NAMES:
        row = fetch_waqi_city(city)
        if row:
            rows.append(row)
        time.sleep(delay)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["data_quality"] = "waqi_live"
    return df
