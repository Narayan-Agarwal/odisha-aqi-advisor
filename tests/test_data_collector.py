"""
Unit and property tests for src/data_collector.py and src/constants.py
"""
import pytest
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data_collector import calculate_aqi
from src.constants import FEATURE_LABELS, TIER_LABELS
from src.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# constants.py
# ---------------------------------------------------------------------------

def test_feature_labels_has_10_keys():
    assert len(FEATURE_LABELS) == 10


def test_feature_labels_keys_match_feature_cols():
    assert set(FEATURE_LABELS.keys()) == set(FEATURE_COLS)


def test_tier_labels_has_3_keys():
    assert len(TIER_LABELS) == 3
    assert set(TIER_LABELS.keys()) == {"heavy_industrial", "urban", "clean_baseline"}


# ---------------------------------------------------------------------------
# calculate_aqi — unit tests
# ---------------------------------------------------------------------------

def test_calculate_aqi_all_none_returns_none():
    assert calculate_aqi(None, None, None, None) is None


def test_calculate_aqi_all_nan_returns_none():
    assert calculate_aqi(float("nan"), float("nan"), float("nan"), float("nan")) is None


def test_calculate_aqi_non_negative():
    result = calculate_aqi(50, 80, 20, 30)
    assert result is not None
    assert result >= 0


def test_calculate_aqi_known_value():
    # pm25=30 → sub_index=50 (at breakpoint boundary)
    result = calculate_aqi(30, None, None, None)
    assert result is not None
    assert abs(result - 50.0) < 0.01


def test_calculate_aqi_single_pollutant_is_max():
    # Only pm10=100 → sub_index=100
    result = calculate_aqi(None, 100, None, None)
    assert result is not None
    assert abs(result - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Property: AQI is always >= 0 for valid inputs
# ---------------------------------------------------------------------------

@given(
    pm25=st.one_of(st.none(), st.floats(min_value=0, max_value=500)),
    pm10=st.one_of(st.none(), st.floats(min_value=0, max_value=600)),
    so2=st.one_of(st.none(), st.floats(min_value=0, max_value=1600)),
    no2=st.one_of(st.none(), st.floats(min_value=0, max_value=400)),
)
@settings(max_examples=200)
def test_aqi_non_negative(pm25, pm10, so2, no2):
    # Feature: odisha-aqi-advisor-v3, Property 3: AQI non-negativity
    result = calculate_aqi(pm25, pm10, so2, no2)
    assert result is None or result >= 0


# ---------------------------------------------------------------------------
# layout_columns threshold (imported from app logic)
# ---------------------------------------------------------------------------

def test_layout_columns_below_768():
    # Feature: odisha-aqi-advisor-v3, Property 14: layout_columns threshold
    from app import layout_columns
    assert layout_columns(0) == 1
    assert layout_columns(767) == 1


def test_layout_columns_at_and_above_768():
    from app import layout_columns
    assert layout_columns(768) == 2
    assert layout_columns(1920) == 2


@given(screen_width=st.integers(min_value=0, max_value=2000))
@settings(max_examples=200)
def test_layout_columns_property(screen_width):
    from app import layout_columns
    result = layout_columns(screen_width)
    if screen_width < 768:
        assert result == 1
    else:
        assert result == 2
