"""
features.py -- Feature engineering for the Odisha AQI Advisor.
All functions are pure transformations (no I/O, no side effects).
Computations are performed per city group to avoid cross-city contamination.
"""
import pandas as pd
import numpy as np
from typing import List

from src.data_loader import CITIES, INDUSTRIAL_CITIES

# Canonical feature column names (must match feature_columns.json)
FEATURE_COLS = [
    "aqi_yesterday",
    "aqi_7day_avg",
    "aqi_3day_avg",
    "aqi_momentum",
    "aqi_3day_change",
    "pm25_lag1",
    "pm25_lag1_sq",
    "pm10_lag1",
    "so2_lag1",
    "no2_lag1",
    "month",
    "is_winter",
    "is_monsoon",
    "is_industrial_peak",
    "winter_industrial",
]

TARGET_COL = "aqi_target"


def add_lag_features(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """Add lag-1 features per city group (sorted by date).

    Adds: aqi_yesterday, pm25_lag1, pm25_lag1_sq, pm10_lag1, so2_lag1, no2_lag1
    All computed within city groups to prevent cross-city contamination.
    """
    df = df.copy()
    df = df.sort_values([city_col, "date"]).reset_index(drop=True)

    df["aqi_yesterday"] = df.groupby(city_col)["aqi"].shift(1)
    df["pm25_lag1"]     = df.groupby(city_col)["pm25"].shift(1)
    df["pm10_lag1"]     = df.groupby(city_col)["pm10"].shift(1)
    df["so2_lag1"]      = df.groupby(city_col)["so2"].shift(1)
    df["no2_lag1"]      = df.groupby(city_col)["no2"].shift(1)
    # Non-linear: PM2.5 squared captures exponential health impact at high concentrations
    df["pm25_lag1_sq"]  = df["pm25_lag1"] ** 2
    return df


def add_rolling_features(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """Add rolling mean features per city group.

    Adds:
    - aqi_7day_avg:   7-day rolling mean (shifted by 1 to avoid leakage)
    - aqi_30day_avg:  30-day rolling mean (shifted by 1, min 7 periods)
    - aqi_3day_avg:   3-day rolling mean (shifted by 1)
    - aqi_momentum:   aqi_yesterday minus aqi_7day_avg (rising/falling indicator)
    - aqi_3day_change: aqi_yesterday minus aqi_3day_avg (short-term rate of change)
    """
    df = df.copy()
    df = df.sort_values([city_col, "date"]).reset_index(drop=True)

    df["aqi_7day_avg"] = df.groupby(city_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["aqi_30day_avg"] = df.groupby(city_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=7).mean()
    )
    df["aqi_3day_avg"] = df.groupby(city_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    # Momentum: positive = AQI rising above recent average, negative = falling
    df["aqi_momentum"]    = df["aqi_yesterday"] - df["aqi_7day_avg"]
    df["aqi_3day_change"] = df["aqi_yesterday"] - df["aqi_3day_avg"]
    return df


def add_target(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """Add aqi_target = next day's AQI per city group."""
    df = df.copy()
    df = df.sort_values([city_col, "date"]).reset_index(drop=True)
    df[TARGET_COL] = df.groupby(city_col)["aqi"].transform(lambda x: x.shift(-1))
    return df


def add_seasonal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, is_winter, is_monsoon flags."""
    df = df.copy()
    df["month"]      = df["date"].dt.month.astype(int)
    df["is_winter"]  = df["month"].isin([11, 12, 1]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    return df


def add_industrial_peak_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_industrial_peak and winter_industrial interaction feature."""
    df = df.copy()
    peak_months = {10, 11, 12, 1, 2}
    df["is_industrial_peak"] = (
        df["city"].isin(INDUSTRIAL_CITIES) & df["date"].dt.month.isin(peak_months)
    ).astype(int)
    # Non-linear interaction: industrial cities in winter are disproportionately worse
    df["winter_industrial"] = (
        df["city"].isin(INDUSTRIAL_CITIES) & df["date"].dt.month.isin({11, 12, 1})
    ).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """Select and order feature columns; drop rows with any NaN in those columns.

    Raises
    ------
    KeyError
        If any column in feature_cols is missing from df.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in DataFrame: {missing}")
    return df[feature_cols].dropna()


def run_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a cleaned DataFrame.

    Steps (in order):
    1. add_lag_features       -- lag-1 pollutant and AQI features (per city)
    2. add_rolling_features   -- 7-day avg, 30-day avg, momentum (per city)
    3. add_target             -- next-day AQI target (per city)
    4. add_seasonal_flags     -- month, winter, monsoon, pre-monsoon
    5. add_industrial_peak_flag
    6. Drop rows where aqi_target or aqi_yesterday is NaN

    Returns the fully featured DataFrame (all original columns + engineered features).
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)
    df = add_seasonal_flags(df)
    df = add_industrial_peak_flag(df)
    df = df.dropna(subset=["aqi_target", "aqi_yesterday"]).reset_index(drop=True)
    return df
