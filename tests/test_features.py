"""
Unit and property tests for src/features.py
Tasks 4.2, 4.3, 4.5, 4.6, 4.7, 4.8
"""
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from src.features import (
    FEATURE_COLS,
    add_diwali_flag,
    add_industrial_peak_flag,
    add_lag_features,
    add_rolling_features,
    add_seasonal_flags,
    build_feature_matrix,
    encode_tier,
    run_full_pipeline,
)
from src.data_loader import CITIES, DIWALI_DATES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_city_df(city="Angul", n=30, start="2021-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "city": city,
        "date": dates,
        "aqi":  np.random.uniform(50, 300, n),
        "pm25": np.random.uniform(10, 150, n),
        "pm10": np.random.uniform(20, 200, n),
        "so2":  np.random.uniform(5, 50, n),
        "no2":  np.random.uniform(5, 80, n),
    })


# ---------------------------------------------------------------------------
# Task 4.8 — Unit tests
# ---------------------------------------------------------------------------

def test_build_feature_matrix_raises_for_missing_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError):
        build_feature_matrix(df, feature_cols=["a", "missing_col"])


def test_encode_tier_maps_all_tiers():
    cities = list(CITIES.keys())
    df = pd.DataFrame({"city": cities, "date": pd.date_range("2021-01-01", periods=len(cities))})
    result = encode_tier(df)
    for city in cities:
        expected = CITIES[city]["tier_code"]
        actual = result.loc[result["city"] == city, "tier_encoded"].iloc[0]
        assert actual == expected, f"{city}: expected {expected}, got {actual}"


def test_add_seasonal_flags_winter():
    df = pd.DataFrame({"date": pd.to_datetime(["2021-12-15", "2021-06-15"])})
    result = add_seasonal_flags(df)
    assert result.loc[0, "is_winter"] == 1
    assert result.loc[1, "is_winter"] == 0


def test_add_seasonal_flags_monsoon():
    df = pd.DataFrame({"date": pd.to_datetime(["2021-08-01", "2021-03-01"])})
    result = add_seasonal_flags(df)
    assert result.loc[0, "is_monsoon"] == 1
    assert result.loc[1, "is_monsoon"] == 0


# ---------------------------------------------------------------------------
# Task 4.2 — Property 5: aqi_lag_1 equals previous day's AQI
# ---------------------------------------------------------------------------

def test_aqi_lag1_equals_previous_day():
    df = _make_city_df(n=20)
    result = add_lag_features(df)
    result = result.sort_values("date").reset_index(drop=True)
    # Row 0 should be NaN (no previous day)
    assert pd.isna(result.loc[0, "aqi_yesterday"])
    # Rows 1+ should equal the previous row's aqi
    for i in range(1, len(result)):
        assert abs(result.loc[i, "aqi_yesterday"] - result.loc[i - 1, "aqi"]) < 1e-9


# ---------------------------------------------------------------------------
# Task 4.3 — Property 6: aqi_rolling_7d_mean correctness
# ---------------------------------------------------------------------------

def test_rolling_7d_mean_correctness():
    df = _make_city_df(n=20)
    result = add_rolling_features(df)
    result = result.sort_values("date").reset_index(drop=True)
    # Row 7: rolling mean of rows 0-6 (shifted by 1, so uses rows before current)
    for i in range(1, len(result)):
        window = result["aqi"].iloc[max(0, i - 7):i].values
        expected = window.mean()
        actual = result.loc[i, "aqi_7day_avg"]
        assert abs(actual - expected) < 1e-6, f"Row {i}: expected {expected:.4f}, got {actual:.4f}"


# ---------------------------------------------------------------------------
# Task 4.5 — Property 7: diwali_week_flag correctness
# ---------------------------------------------------------------------------

def test_diwali_flag_within_3_days():
    diwali_ts = pd.Timestamp(DIWALI_DATES[0])
    dates_in = [diwali_ts + pd.Timedelta(days=d) for d in range(-3, 4)]
    dates_out = [diwali_ts + pd.Timedelta(days=d) for d in [-4, 4, 10, -10]]
    df_in = pd.DataFrame({"date": dates_in})
    df_out = pd.DataFrame({"date": dates_out})
    assert add_diwali_flag(df_in)["is_diwali_week"].eq(1).all()
    assert add_diwali_flag(df_out)["is_diwali_week"].eq(0).all()


# ---------------------------------------------------------------------------
# Task 4.6 — Property 8: industrial_peak_flag correctness
# ---------------------------------------------------------------------------

def test_industrial_peak_flag():
    peak_months = [10, 11, 12, 1, 2]
    non_peak_months = [3, 4, 5, 6, 7, 8, 9]
    industrial_city = "Angul"
    urban_city = "Bhubaneswar"

    rows = []
    for m in peak_months:
        rows.append({"city": industrial_city, "date": pd.Timestamp(f"2021-{m:02d}-15")})
        rows.append({"city": urban_city,      "date": pd.Timestamp(f"2021-{m:02d}-15")})
    for m in non_peak_months:
        rows.append({"city": industrial_city, "date": pd.Timestamp(f"2021-{m:02d}-15")})

    df = pd.DataFrame(rows)
    result = add_industrial_peak_flag(df)

    # Industrial city in peak months → 1
    ind_peak = result[(result["city"] == industrial_city) & (result["date"].dt.month.isin(peak_months))]
    assert ind_peak["is_industrial_peak"].eq(1).all()

    # Urban city in peak months → 0
    urb_peak = result[(result["city"] == urban_city) & (result["date"].dt.month.isin(peak_months))]
    assert urb_peak["is_industrial_peak"].eq(0).all()

    # Industrial city in non-peak months → 0
    ind_nonpeak = result[(result["city"] == industrial_city) & (result["date"].dt.month.isin(non_peak_months))]
    assert ind_nonpeak["is_industrial_peak"].eq(0).all()


# ---------------------------------------------------------------------------
# Task 4.7 — Property 4: Feature engineering produces correct schema, no NaN
# ---------------------------------------------------------------------------

def test_run_full_pipeline_schema_and_no_nan():
    df = _make_city_df(n=60)
    result = run_full_pipeline(df)
    # All FEATURE_COLS present
    for col in FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"
    # No NaN in feature columns
    nan_counts = result[FEATURE_COLS].isna().sum()
    assert nan_counts.sum() == 0, f"NaN found: {nan_counts[nan_counts > 0].to_dict()}"
