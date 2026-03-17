"""
constants.py — Centralised label mappings for the Odisha AQI Advisor.
"""

FEATURE_LABELS = {
    "aqi_yesterday":      "Yesterday's AQI",
    "aqi_7day_avg":       "7-Day Avg AQI",
    "pm25_lag1":          "PM2.5 (prev day)",
    "pm10_lag1":          "PM10 (prev day)",
    "so2_lag1":           "SO₂ (prev day)",
    "no2_lag1":           "NO₂ (prev day)",
    "month":              "Month",
    "is_winter":          "Winter Season",
    "is_monsoon":         "Monsoon Season",
    "is_industrial_peak": "Industrial Peak",
}

TIER_LABELS = {
    "heavy_industrial": "Heavy Industrial",
    "urban":            "Urban",
    "clean_baseline":   "Clean Baseline",
}

# CPCB AQI category bands: (lo, hi, hex_colour, label)
AQI_BANDS = [
    (0,   50,  "#00B050", "Good"),
    (51,  100, "#92D050", "Satisfactory"),
    (101, 200, "#FFFF00", "Moderate"),
    (201, 300, "#FF7C00", "Poor"),
    (301, 400, "#FF0000", "Very Poor"),
    (401, 500, "#7B0023", "Severe"),
]
