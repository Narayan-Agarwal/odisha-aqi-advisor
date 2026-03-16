"""
Unit tests for src/visualisations.py
Tasks 7.2, 7.3
"""
import glob
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.visualisations import (
    plot_city_month_heatmap,
    plot_diwali_spike,
    plot_industrial_vs_urban,
    plot_monsoon_dip,
    plot_tier_comparison,
    plot_yoy_trend,
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


# ---------------------------------------------------------------------------
# Task 7.3 — Each chart function returns a valid .png path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    return _make_df()


@pytest.fixture()
def tmp_charts(tmp_path):
    return str(tmp_path)


def test_plot_tier_comparison_returns_png(sample_df, tmp_charts):
    path = plot_tier_comparison(sample_df, out_dir=tmp_charts)
    assert path.endswith(".png")
    assert os.path.exists(path)


def test_plot_city_month_heatmap_returns_png(sample_df, tmp_charts):
    path = plot_city_month_heatmap(sample_df, out_dir=tmp_charts)
    assert path.endswith(".png")
    assert os.path.exists(path)


def test_plot_monsoon_dip_returns_png(sample_df, tmp_charts):
    path = plot_monsoon_dip(sample_df, out_dir=tmp_charts)
    assert path.endswith(".png")
    assert os.path.exists(path)


def test_plot_yoy_trend_returns_png(sample_df, tmp_charts):
    path = plot_yoy_trend(sample_df, out_dir=tmp_charts)
    assert path.endswith(".png")
    assert os.path.exists(path)


def test_plot_diwali_spike_returns_png(sample_df, tmp_charts):
    path = plot_diwali_spike(sample_df, out_dir=tmp_charts)
    assert path.endswith(".png")
    assert os.path.exists(path)


def test_plot_industrial_vs_urban_returns_png(sample_df, tmp_charts):
    path = plot_industrial_vs_urban(sample_df, out_dir=tmp_charts)
    assert path.endswith(".png")
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Task 7.2 — Property 16: Chart generation is idempotent
# ---------------------------------------------------------------------------

def test_chart_idempotent(sample_df, tmp_charts):
    """Calling the same chart function twice produces identical file sizes."""
    path1 = plot_tier_comparison(sample_df, out_dir=tmp_charts)
    size1 = os.path.getsize(path1)
    path2 = plot_tier_comparison(sample_df, out_dir=tmp_charts)
    size2 = os.path.getsize(path2)
    assert path1 == path2
    # File should be overwritten (same path), size within 5% tolerance
    assert abs(size1 - size2) / max(size1, 1) < 0.05


# ---------------------------------------------------------------------------
# Task 7.3 — Verify charts/ directory has expected PNGs after pipeline
# ---------------------------------------------------------------------------

def test_charts_directory_has_pngs():
    """After running the offline pipeline, charts/ should have PNG files."""
    pngs = glob.glob("charts/*.png")
    assert len(pngs) > 0, "No PNG files found in charts/ — run notebooks 03 and 04 first"
