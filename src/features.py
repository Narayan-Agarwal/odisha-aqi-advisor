"""
features.py — Feature engineering for the Odisha AQI Advisor.
All functions are pure transformations (no I/O, no side effects).
Computations are performed per city group to avoid cross-city contamination.
"""
import pandas as pd
import numpy as np
from typing import List

from src.data_loader import CITIES, INDUSTRIAL_CITIES, DIWALI_DATES

# Canonical feature column names (must match feature_columns.json)
FEATURE_COLS = [
    "aqi_yesterday",
    "aqi_7day_avg",
    "pm25_lag1",
    "pm10_lag1",
    "so2_lag1",
    "no2_lag1",
    "month",
    "is_winter",
    "is_monsoon",
    "is_industrial_peak",
]

TARGET_COL = "aqi_target"


def add_lag_features(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """Add lag-1 features per city group (sorted by date).

    Adds: aqi_yesterday, pm25_lag1, pm10_lag1, so2_lag1, no2_lag1
    """
    df = df.copy()
    df = df.sort_values([city_col, "date"]).reset_index(drop=True)

    def _lag1(series):
        return series.shift(1)

    df["aqi_yesterday"] = df.groupby(city_col)["aqi"].transform(_lag1)
    df["pm25_lag1"]     = df.groupby(city_col)["pm25"].transform(_lag1)
    df["pm10_lag1"]     = df.groupby(city_col)["pm10"].transform(_lag1)
    df["so2_lag1"]      = df.groupby(city_col)["so2"].transform(_lag1)
    df["no2_lag1"]      = df.groupby(city_col)["no2"].transform(_lag1)
    return df


def add_rolling_features(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """Add rolling mean features per city group.

    Adds: aqi_7day_avg (7-day rolling mean of aqi, shifted by 1 to avoid leakage)
    """
    df = df.copy()
    df = df.sort_values([city_col, "date"]).reset_index(drop=True)

    def _rolling7(series):
        return series.shift(1).rolling(7, min_periods=1).mean()

    df["aqi_7day_avg"] = df.groupby(city_col)["aqi"].transform(_rolling7)
    return df


def add_target(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """Add aqi_target = next day's AQI per city group."""
    df = df.copy()
    df = df.sort_values([city_col, "date"]).reset_index(drop=True)

    def _shift_minus1(series):
        return series.shift(-1)

    df[TARGET_COL] = df.groupby(city_col)["aqi"].transform(_shift_minus1)
    return df


def add_seasonal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, is_winter, is_monsoon flags.

    is_winter: 1 if month in {11, 12, 1} else 0
    is_monsoon: 1 if month in {7, 8, 9} else 0
    """
    df = df.copy()
    df["month"]      = df["date"].dt.month.astype(int)
    df["is_winter"]  = df["month"].isin([11, 12, 1]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    return df


def add_diwali_flag(df: pd.DataFrame, diwali_dates: List[str] = None) -> pd.DataFrame:
    """Add is_diwali_week: 1 if date within ±3 days of any Diwali date, else 0."""
    if diwali_dates is None:
        diwali_dates = DIWALI_DATES
    df = df.copy()
    diwali_ts = [pd.Timestamp(d) for d in diwali_dates]

    def _is_diwali(date):
        return int(any(abs((date - d).days) <= 3 for d in diwali_ts))

    df["is_diwali_week"] = df["date"].apply(_is_diwali)
    return df


def add_industrial_peak_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_industrial_peak: 1 if city in INDUSTRIAL_CITIES and month in {10,11,12,1,2}."""
    df = df.copy()
    peak_months = {10, 11, 12, 1, 2}
    df["is_industrial_peak"] = (
        df["city"].isin(INDUSTRIAL_CITIES) & df["date"].dt.month.isin(peak_months)
    ).astype(int)
    return df


def encode_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Add tier_encoded: integer tier code from CITIES dict."""
    df = df.copy()
    df["tier_encoded"] = df["city"].map(lambda c: CITIES.get(c, {}).get("tier_code", 0))
    return df


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """Select and order feature columns; drop rows with any NaN in those columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all feature columns present.
    feature_cols : list[str], optional
        Ordered list of feature column names. Defaults to FEATURE_COLS.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the specified columns, no NaN rows.

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
    result = df[feature_cols].dropna()
    return result


def run_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a cleaned DataFrame.

    Steps (in order):
    1. add_lag_features
    2. add_rolling_features
    3. add_target
    4. add_seasonal_flags
    5. add_industrial_peak_flag
    6. Drop rows where aqi_target or aqi_yesterday is NaN

    Returns the fully featured DataFrame (all original columns + engineered features).
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)
    df = add_seasonal_flags(df)
    df = add_industrial_peak_flag(df)
    # Drop rows where target or primary lag is NaN (start/end of each city series)
    df = df.dropna(subset=["aqi_target", "aqi_yesterday"]).reset_index(drop=True)
    return df
