"""
Unit and property tests for src/model.py
Tasks 5.2, 5.3, 5.4
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model import (
    chronological_split,
    evaluate_model,
    load_and_verify,
    save_model,
    train_linear,
    train_xgboost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "y":  np.random.randn(n),
    })


# ---------------------------------------------------------------------------
# Task 5.4 — Unit tests
# ---------------------------------------------------------------------------

def test_chronological_split_raises_for_unsorted():
    df = _make_df(20)
    df = df.iloc[::-1].reset_index(drop=True)  # reverse = descending
    with pytest.raises(ValueError):
        chronological_split(df)


def test_chronological_split_80_20_ratio():
    df = _make_df(100)
    train, test = chronological_split(df, test_fraction=0.2)
    assert len(train) == 80
    assert len(test) == 20


def test_evaluate_model_returns_correct_keys():
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    model = train_linear(X, y)
    metrics = evaluate_model(model, X, y)
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_train_linear_returns_fitted_estimator():
    X = np.random.randn(40, 3)
    y = np.random.randn(40)
    model = train_linear(X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert preds.shape == (40,)


def test_train_xgboost_returns_fitted_estimator():
    X = np.random.randn(40, 3)
    y = np.random.randn(40)
    model = train_xgboost(X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert preds.shape == (40,)


# ---------------------------------------------------------------------------
# Task 5.2 — Property 9: Chronological split preserves time order + 80/20
# ---------------------------------------------------------------------------

@given(st.integers(min_value=10, max_value=200))
@settings(max_examples=50)
def test_chronological_split_preserves_order(n):
    df = _make_df(n)
    train, test = chronological_split(df, test_fraction=0.2)
    # Train dates all before test dates
    if len(train) > 0 and len(test) > 0:
        assert train["date"].max() <= test["date"].min()
    # Correct split size
    expected_train = int(n * 0.8)
    assert len(train) == expected_train
    assert len(train) + len(test) == n


# ---------------------------------------------------------------------------
# Task 5.3 — Property 10: Model serialisation round-trip preserves predictions
# ---------------------------------------------------------------------------

def test_model_serialisation_roundtrip_linear():
    X = np.random.randn(30, 4)
    y = np.random.randn(30)
    model = train_linear(X, y)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model(model, path)
        assert load_and_verify(model, path, X, tol=1e-5)
    finally:
        os.unlink(path)


def test_model_serialisation_roundtrip_xgboost():
    X = np.random.randn(30, 4)
    y = np.random.randn(30)
    model = train_xgboost(X, y)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model(model, path)
        assert load_and_verify(model, path, X, tol=1e-5)
    finally:
        os.unlink(path)
