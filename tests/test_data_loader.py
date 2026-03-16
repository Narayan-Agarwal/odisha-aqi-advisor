"""
Unit tests for src/data_loader.py
Tasks 2.2, 2.3, 2.4
"""
import json
import os
import tempfile

import pandas as pd
import pytest

from src.data_loader import load_feature_columns, load_model, load_model_results


# ---------------------------------------------------------------------------
# Task 2.4 — Unit tests
# ---------------------------------------------------------------------------

def test_load_model_raises_for_nonexistent_city():
    """load_model raises FileNotFoundError for a city with no .joblib file."""
    with pytest.raises(FileNotFoundError):
        load_model("NonExistentCity", "xgb", models_dir="models")


def test_load_feature_columns_returns_list_of_strings():
    """load_feature_columns returns a non-empty list of strings in correct order."""
    path = "models/feature_columns.json"
    if not os.path.exists(path):
        pytest.skip("feature_columns.json not present — run notebook 04 first")
    cols = load_feature_columns(path)
    assert isinstance(cols, list)
    assert len(cols) > 0
    assert all(isinstance(c, str) for c in cols)


def test_load_feature_columns_correct_order():
    """load_feature_columns preserves the order written to disk."""
    path = "models/feature_columns.json"
    if not os.path.exists(path):
        pytest.skip("feature_columns.json not present — run notebook 04 first")
    cols = load_feature_columns(path)
    with open(path) as f:
        raw = json.load(f)
    assert cols == raw


def test_load_model_results_raises_when_absent():
    """load_model_results raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        load_model_results(path="/nonexistent/path/model_results.csv")


# ---------------------------------------------------------------------------
# Task 2.2 — Property 15: CSV round-trip preserves DataFrame equality
# ---------------------------------------------------------------------------

def test_csv_roundtrip_preserves_dataframe():
    """Writing a DataFrame to CSV and reading it back preserves values."""
    df = pd.DataFrame({
        "city": ["Angul", "Bhubaneswar"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "aqi": [120.5, 85.0],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        df.to_csv(path, index=False)
        df2 = pd.read_csv(path, parse_dates=["date"])
        pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Task 2.3 — Property 11: feature_columns.json round-trip preserves order
# ---------------------------------------------------------------------------

def test_feature_columns_json_roundtrip_preserves_order():
    """Writing a list to JSON and reading it back preserves order."""
    cols = ["aqi_yesterday", "aqi_7day_avg", "pm25_lag1", "month", "tier_encoded"]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(cols, f)
        path = f.name
    try:
        result = load_feature_columns(path)
        assert result == cols
    finally:
        os.unlink(path)
