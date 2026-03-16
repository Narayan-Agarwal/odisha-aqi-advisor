"""
Property tests for data cleaning invariants (notebook 01 outputs).
Tasks 8.1, 8.2, 8.3
"""
import os

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


CLEAN_CSV = "data/processed/unified_clean.csv"


@pytest.fixture(scope="module")
def clean_df():
    if not os.path.exists(CLEAN_CSV):
        pytest.skip("unified_clean.csv not present — run notebook 01 first")
    return pd.read_csv(CLEAN_CSV, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Task 8.1 — Property 1: No NaN AQI after cleaning
# ---------------------------------------------------------------------------

def test_no_nan_aqi(clean_df):
    assert clean_df["aqi"].isna().sum() == 0, "Found NaN values in aqi column"


# ---------------------------------------------------------------------------
# Task 8.2 — Property 2: No duplicate (city, date) pairs
# ---------------------------------------------------------------------------

def test_no_duplicate_city_date(clean_df):
    dupes = clean_df.duplicated(subset=["city", "date"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (city, date) pairs"


# ---------------------------------------------------------------------------
# Task 8.3 — Property 3: Date range filter — all dates within 2019-2023
# ---------------------------------------------------------------------------

def test_date_range_within_bounds(clean_df):
    min_date = pd.Timestamp("2019-01-01")
    max_date = pd.Timestamp("2023-12-31")
    assert clean_df["date"].min() >= min_date, f"Dates before 2019-01-01 found"
    assert clean_df["date"].max() <= max_date, f"Dates after 2023-12-31 found"
