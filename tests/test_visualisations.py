"""
Unit tests for src/visualisations.py (V3 — Plotly)
"""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.visualisations import (
    plot_city_month_heatmap,
    plot_diwali_spike,
    plot_industrial_vs_urban,
    plot_monsoon_dip,
    plot_tier_comparison,
    plot_yoy_trend,
    plot_industrial_corridor,
    plot_pollutant_correlation,
    plot_pollutant_dominance,
    plot_model_comparison,
    plot_historical_aqi,
    plot_feature_importance_city,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df():
    from src.data_loader import CITIES
    rows = []
    for city in CITIES:
        for year in [2021, 2022]:
            for month in range(1, 13):
                rows.append({
                    "city": city,
                    "date": pd.Timestamp(f"{year}-{month:02d}-15"),
                    "aqi": np.random.uniform(50, 300),
                    "pm25": np.random.uniform(10, 100),
                    "pm10": np.random.uniform(20, 150),
                    "no2": np.random.uniform(5, 60),
                    "so2": np.random.uniform(5, 40),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sample_df():
    return _make_df()


# ---------------------------------------------------------------------------
# All chart functions return go.Figure
# ---------------------------------------------------------------------------

def test_plot_tier_comparison_returns_figure(sample_df):
    fig = plot_tier_comparison(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_city_month_heatmap_returns_figure(sample_df):
    fig = plot_city_month_heatmap(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_monsoon_dip_returns_figure(sample_df):
    fig = plot_monsoon_dip(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_yoy_trend_returns_figure(sample_df):
    fig = plot_yoy_trend(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_diwali_spike_returns_figure(sample_df):
    fig = plot_diwali_spike(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_industrial_vs_urban_returns_figure(sample_df):
    fig = plot_industrial_vs_urban(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_industrial_corridor_returns_figure(sample_df):
    fig = plot_industrial_corridor(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_pollutant_correlation_returns_figure(sample_df):
    fig = plot_pollutant_correlation(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_pollutant_dominance_returns_figure(sample_df):
    fig = plot_pollutant_dominance(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_historical_aqi_monthly_returns_figure(sample_df):
    city = "Angul"
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2022-12-31")
    fig = plot_historical_aqi(sample_df, city, start, end, granularity="monthly")
    assert isinstance(fig, go.Figure)


def test_plot_historical_aqi_daily_returns_figure(sample_df):
    city = "Angul"
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2022-12-31")
    fig = plot_historical_aqi(sample_df, city, start, end, granularity="daily")
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# All traces have non-empty hovertemplate
# ---------------------------------------------------------------------------

def _all_traces_have_hovertemplate(fig: go.Figure) -> bool:
    for trace in fig.data:
        ht = getattr(trace, "hovertemplate", None)
        if not ht:
            return False
    return True


def test_tier_comparison_hovertemplates(sample_df):
    fig = plot_tier_comparison(sample_df)
    assert _all_traces_have_hovertemplate(fig)


def test_heatmap_hovertemplates(sample_df):
    fig = plot_city_month_heatmap(sample_df)
    assert _all_traces_have_hovertemplate(fig)


def test_historical_aqi_hovertemplates(sample_df):
    fig = plot_historical_aqi(sample_df, "Angul",
                              pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"))
    assert _all_traces_have_hovertemplate(fig)


# ---------------------------------------------------------------------------
# Height parameter is respected
# ---------------------------------------------------------------------------

def test_chart_height_250(sample_df):
    fig = plot_tier_comparison(sample_df, height=250)
    assert fig.layout.height == 250


def test_chart_height_380(sample_df):
    fig = plot_monsoon_dip(sample_df, height=380)
    assert fig.layout.height == 380


# ---------------------------------------------------------------------------
# Model comparison chart
# ---------------------------------------------------------------------------

def test_plot_model_comparison_returns_figure():
    results = pd.DataFrame({
        "city": ["Angul", "Angul", "Bhubaneswar", "Bhubaneswar"],
        "model_type": ["lr", "xgb", "lr", "xgb"],
        "mae": [40.0, 38.0, 25.0, 24.0],
        "rmse": [50.0, 48.0, 32.0, 30.0],
        "r2": [0.4, 0.45, 0.3, 0.35],
    })
    fig = plot_model_comparison(results)
    assert isinstance(fig, go.Figure)
