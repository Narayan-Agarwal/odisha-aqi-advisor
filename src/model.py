"""
model.py — Model training, evaluation, and serialisation for the Odisha AQI Advisor.
"""
import math
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def chronological_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train and test sets chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date ascending. Must contain a 'date' column.
    test_fraction : float
        Fraction of rows to use as test set (default 0.2).

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        train_df contains the first floor(n * (1 - test_fraction)) rows.
        test_df contains the remaining rows.

    Raises
    ------
    ValueError
        If df is not sorted by date ascending.
    """
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        if not dates.is_monotonic_increasing:
            raise ValueError(
                "DataFrame must be sorted by date ascending for chronological split."
            )
    n = len(df)
    split_idx = int(n * (1.0 - test_fraction))
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    return train_df, test_df


def train_linear(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit and return a LinearRegression model with all defaults."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """Fit and return an XGBRegressor with canonical hyperparameters."""
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate a fitted model and return MAE, RMSE, R².

    Returns
    -------
    dict with keys: 'mae', 'rmse', 'r2'
    """
    y_pred = model.predict(X_test)
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(model: Any, path: str) -> None:
    """Serialise a fitted model to path using joblib.dump.

    Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(model, path)


def load_and_verify(
    model: Any, path: str, X_test: np.ndarray, tol: float = 1e-5
) -> bool:
    """Load a model from path and verify predictions match the in-memory model.

    Parameters
    ----------
    model : fitted estimator
        The in-memory model whose predictions are the reference.
    path : str
        Path to the .joblib file.
    X_test : np.ndarray
        Feature matrix for comparison.
    tol : float
        Maximum allowed absolute difference per prediction (default 1e-5).

    Returns
    -------
    bool
        True if all predictions match within tol, False otherwise.
    """
    loaded = joblib.load(path)
    preds_original = model.predict(X_test)
    preds_loaded   = loaded.predict(X_test)
    return bool(np.allclose(preds_original, preds_loaded, atol=tol))
