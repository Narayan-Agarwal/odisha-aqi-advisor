"""
auto_updater.py — Fetch latest WAQI data and append to featured.csv.

Called once per Streamlit session at app startup. Never crashes the app.
"""
import logging
import os
from datetime import datetime

import pandas as pd
import requests

log = logging.getLogger(__name__)

WAQI_TOKEN = "ef5feda541fb94c36eb71257d7687ef36bc929ca"

WAQI_CITY_NAMES = {
    "Angul":       ["angul",       "india/angul"],
    "Talcher":     ["talcher",     "india/talcher"],
    "Rourkela":    ["rourkela",    "india/rourkela"],
    "Sambalpur":   ["sambalpur",   "india/sambalpur"],
    "Jharsuguda":  ["jharsuguda",  "india/jharsuguda"],
    "Bhubaneswar": ["bhubaneswar", "india/bhubaneswar"],
    "Cuttack":     ["cuttack",     "india/cuttack"],
    "Balasore":    ["balasore",    "india/balasore"],
    "Berhampur":   ["brahmapur",   "india/brahmapur", "berhampur", "india/berhampur"],
    "Rayagada":    ["rayagada",    "india/rayagada"],
}

FEATURED_PATH = "data/processed/featured.csv"
LOG_PATH      = "data/processed/waqi_fetch_log.txt"


def _fetch_station(slug: str):
    """Fetch WAQI feed for one station slug. Returns data dict or None."""
    try:
        url = f"https://api.waqi.info/feed/{slug}/"
        resp = requests.get(url, params={"token": WAQI_TOKEN}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "ok":
            return payload["data"]
    except Exception as exc:
        log.debug("WAQI fetch failed for %s: %s", slug, exc)
    return None


def _extract(data: dict, key: str):
    try:
        return float(data["iaqi"][key]["v"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_latest_waqi(city_name: str):
    """Try all slugs for a city. Returns a row dict or None."""
    for slug in WAQI_CITY_NAMES.get(city_name, [city_name.lower()]):
        data = _fetch_station(slug)
        if data is None:
            continue
        try:
            aqi_val = float(data["aqi"])
        except (KeyError, TypeError, ValueError):
            continue

        raw_time = data.get("time", {}).get("s", "")
        try:
            obs_date = pd.Timestamp(raw_time).strftime("%Y-%m-%d")
        except Exception:
            obs_date = datetime.today().strftime("%Y-%m-%d")

        return {
            "date":  obs_date,
            "city":  city_name,
            "aqi":   round(aqi_val, 2),
            "pm25":  _extract(data, "pm25"),
            "pm10":  _extract(data, "pm10"),
            "no2":   _extract(data, "no2"),
            "so2":   _extract(data, "so2"),
            "o3":    _extract(data, "o3"),
            "co":    _extract(data, "co"),
        }
    return None


def compute_features_for_new_row(new_row: dict, city_df: pd.DataFrame) -> dict:
    """Compute the 10 engineered features using the city's existing history."""
    city_df = city_df.sort_values("date")
    last = city_df.iloc[-1]
    date = pd.to_datetime(new_row["date"])
    month = date.month

    new_row["aqi_yesterday"]      = float(last["aqi"]) if pd.notna(last["aqi"]) else city_df["aqi"].median()
    new_row["aqi_7day_avg"]       = float(city_df["aqi"].tail(7).mean())
    new_row["pm25_lag1"]          = float(last["pm25"]) if pd.notna(last.get("pm25")) else float(city_df["pm25"].median())
    new_row["pm10_lag1"]          = float(last["pm10"]) if pd.notna(last.get("pm10")) else float(city_df["pm10"].median())
    new_row["so2_lag1"]           = float(last["so2"])  if pd.notna(last.get("so2"))  else float(city_df["so2"].median())
    new_row["no2_lag1"]           = float(last["no2"])  if pd.notna(last.get("no2"))  else float(city_df["no2"].median())
    new_row["month"]              = month
    new_row["is_winter"]          = 1 if month in [11, 12, 1] else 0
    new_row["is_monsoon"]         = 1 if month in [7, 8, 9]   else 0
    new_row["is_industrial_peak"] = 1 if month in [10, 11, 12, 1, 2] else 0
    return new_row


def run_auto_update():
    """
    Fetch latest WAQI data for all cities and append new rows to featured.csv.
    Safe to call at app startup — never raises exceptions.
    """
    if not os.path.exists(FEATURED_PATH):
        log.warning("featured.csv not found — skipping auto-update")
        return

    try:
        featured_df = pd.read_csv(FEATURED_PATH)
        featured_df["date"] = pd.to_datetime(featured_df["date"]).dt.strftime("%Y-%m-%d")
    except Exception as exc:
        log.error("Failed to load featured.csv: %s", exc)
        return

    new_rows  = []
    log_lines = [f"=== Auto-update run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="]

    for city in WAQI_CITY_NAMES:
        try:
            result = fetch_latest_waqi(city)

            if result is None:
                log_lines.append(f"SKIP  {city}: WAQI returned no data")
                continue

            date_str = result["date"]

            # Skip if already in dataset
            already_exists = (
                (featured_df["city"] == city) &
                (featured_df["date"] == date_str)
            ).any()

            if already_exists:
                log_lines.append(f"SKIP  {city}: {date_str} already in dataset")
                continue

            # Need at least 7 rows of history to compute features
            city_df = featured_df[featured_df["city"] == city].copy()
            if len(city_df) < 7:
                log_lines.append(f"SKIP  {city}: insufficient history ({len(city_df)} rows)")
                continue

            result = compute_features_for_new_row(result, city_df)
            new_rows.append(result)
            log_lines.append(f"ADD   {city}: {date_str} — AQI {result['aqi']}")

        except Exception as exc:
            log_lines.append(f"ERROR {city}: {exc}")
            log.error("Auto-update error for %s: %s", city, exc)

    if new_rows:
        try:
            new_df = pd.DataFrame(new_rows)
            featured_df = pd.concat([featured_df, new_df], ignore_index=True)
            featured_df = featured_df.sort_values(["city", "date"]).reset_index(drop=True)
            featured_df.to_csv(FEATURED_PATH, index=False)
            log_lines.append(f"SAVED {len(new_rows)} new rows to featured.csv")
        except Exception as exc:
            log_lines.append(f"ERROR saving featured.csv: {exc}")
            log.error("Failed to save featured.csv: %s", exc)
    else:
        log_lines.append("No new rows added — dataset already up to date")

    # Append to log file
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n\n")
    except Exception:
        pass
