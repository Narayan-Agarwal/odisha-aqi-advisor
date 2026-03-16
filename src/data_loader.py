"""
data_loader.py — Constants and I/O utilities for the Odisha AQI Advisor.
All file paths are relative to the project root (odisha-aqi-advisor/).
"""
import json
import os
from typing import Any

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# City reference data (case-sensitive keys)
# ---------------------------------------------------------------------------
CITIES = {
    'Jharsuguda':  {'tier': 'heavy_industrial', 'tier_code': 2, 'district': 'Jharsuguda',
                    'industry': 'Vedanta aluminium smelter, JSPL thermal power'},
    'Angul':       {'tier': 'heavy_industrial', 'tier_code': 2, 'district': 'Angul',
                    'industry': 'NALCO aluminium (worlds largest single site), NTPC Kaniha'},
    'Talcher':     {'tier': 'heavy_industrial', 'tier_code': 2, 'district': 'Angul',
                    'industry': 'Talcher coalfields, NTPC Talcher, coal transport dust'},
    'Rourkela':    {'tier': 'heavy_industrial', 'tier_code': 2, 'district': 'Sundargarh',
                    'industry': 'SAIL steel plant, fertiliser and chemical industry'},
    'Sambalpur':   {'tier': 'heavy_industrial', 'tier_code': 2, 'district': 'Sambalpur',
                    'industry': 'Cement plants, coal belt proximity, road dust'},
    'Bhubaneswar': {'tier': 'urban',            'tier_code': 1, 'district': 'Khordha',
                    'industry': 'Traffic, construction activity, urban growth'},
    'Cuttack':     {'tier': 'urban',            'tier_code': 1, 'district': 'Cuttack',
                    'industry': 'Dense traffic, old city road network, biomass burning'},
    'Bhadrak':     {'tier': 'urban',            'tier_code': 1, 'district': 'Bhadrak',
                    'industry': 'Port-adjacent industry, light manufacturing'},
    'Ganjam':      {'tier': 'clean_baseline',   'tier_code': 0, 'district': 'Ganjam',
                    'industry': 'Coastal city, sea breeze dispersion, light industry'},
    'Koraput':     {'tier': 'clean_baseline',   'tier_code': 0, 'district': 'Koraput',
                    'industry': 'Hilly terrain, bauxite mining, lower density'},
}

TIER_COLOURS = {
    'heavy_industrial': '#E74C3C',
    'urban':            '#F39C12',
    'clean_baseline':   '#27AE60',
}

INDUSTRIAL_CITIES = ['Jharsuguda', 'Angul', 'Talcher', 'Rourkela', 'Sambalpur']
URBAN_CITIES      = ['Bhubaneswar', 'Cuttack', 'Bhadrak']
CLEAN_CITIES      = ['Ganjam', 'Koraput']
CORRIDOR_CITIES   = ['Jharsuguda', 'Angul', 'Talcher']

DIWALI_DATES = ['2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12']

# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------

def load_featured_csv(path: str = "data/processed/featured.csv") -> pd.DataFrame:
    """Load featured.csv. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"featured.csv not found at '{path}'. Run notebooks 01-03 first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_model_results(path: str = "data/processed/model_results.csv") -> pd.DataFrame:
    """Load model_results.csv. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"model_results.csv not found at '{path}'. Run notebook 04 first."
        )
    return pd.read_csv(path)


def load_model(city: str, model_type: str, models_dir: str = "models") -> Any:
    """Load a fitted estimator from models/{model_type}_{city_lower}.joblib.

    Parameters
    ----------
    city : str
        City name (case-insensitive; will be lowercased).
    model_type : str
        'xgb' or 'lr'.
    models_dir : str
        Directory containing .joblib files.

    Raises
    ------
    FileNotFoundError
        If the .joblib file does not exist.
    """
    city_slug = city.lower()
    filename = f"{model_type}_{city_slug}.joblib"
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'. Run notebook 04 first."
        )
    return joblib.load(path)


def load_feature_columns(path: str = "models/feature_columns.json") -> list:
    """Load and return the ordered feature column list from feature_columns.json.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the file is empty or does not contain a non-empty list of strings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"feature_columns.json not found at '{path}'. Run notebook 04 first."
        )
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not cols or not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError(
            f"feature_columns.json at '{path}' must contain a non-empty list of strings."
        )
    return cols
