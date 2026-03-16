"""
Unit and property tests for src/advisory.py
Tasks 3.2, 3.3, 3.4, 3.5
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.advisory import AQI_BANDS, get_advisory


# ---------------------------------------------------------------------------
# Task 3.5 — Unit tests: boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aqi,expected_cat,expected_colour", [
    (0,   "Good",      "#00B050"),
    (50,  "Good",      "#00B050"),
    (51,  "Satisfactory", "#92D050"),
    (100, "Satisfactory", "#92D050"),
    (101, "Moderate",  "#FFD700"),
    (200, "Moderate",  "#FFD700"),
    (201, "Poor",      "#FF7C00"),
    (300, "Poor",      "#FF7C00"),
    (301, "Very Poor", "#FF0000"),
    (400, "Very Poor", "#FF0000"),
    (401, "Severe",    "#7B0023"),
    (999, "Severe",    "#7B0023"),
])
def test_get_advisory_boundary_values(aqi, expected_cat, expected_colour):
    cat, msg, colour = get_advisory(aqi)
    assert cat == expected_cat
    assert colour == expected_colour
    assert isinstance(msg, str) and len(msg) > 0


def test_get_advisory_negative_raises():
    with pytest.raises(ValueError):
        get_advisory(-1)


def test_get_advisory_non_numeric_raises():
    with pytest.raises(ValueError):
        get_advisory("bad")


# ---------------------------------------------------------------------------
# Task 3.2 — Property 12: get_advisory maps AQI values to correct categories
# ---------------------------------------------------------------------------

@given(st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_advisory_maps_to_correct_band(aqi):
    cat, msg, colour = get_advisory(aqi)
    # Find expected band
    for lower, upper, expected_cat, _, expected_colour in AQI_BANDS:
        if lower <= aqi < upper:
            assert cat == expected_cat
            assert colour == expected_colour
            return
    # If no band matched (shouldn't happen for 0-500), fallback is Severe
    assert cat == "Severe"


# ---------------------------------------------------------------------------
# Task 3.3 — Property 13: get_advisory raises ValueError for invalid input
# ---------------------------------------------------------------------------

@given(st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_advisory_raises_for_negative(aqi):
    with pytest.raises(ValueError):
        get_advisory(aqi)


# ---------------------------------------------------------------------------
# Task 3.4 — Property 14: get_advisory is idempotent
# ---------------------------------------------------------------------------

@given(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_advisory_is_idempotent(aqi):
    result1 = get_advisory(aqi)
    result2 = get_advisory(aqi)
    assert result1 == result2
