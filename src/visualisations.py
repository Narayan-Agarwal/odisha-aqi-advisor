"""
visualisations.py — Chart generation for the Odisha AQI Advisor.
All functions save PNG files to the charts/ directory at >=150 DPI.
"""
import os
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import (
    CITIES, TIER_COLOURS, INDUSTRIAL_CITIES, CORRIDOR_CITIES, DIWALI_DATES
)

sns.set_style("whitegrid")
DPI = 150


def _save(fig: plt.Figure, path: str) -> str:
    """Save figure to path at DPI, close it, return path."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _city_colour(city: str) -> str:
    tier = CITIES.get(city, {}).get("tier", "urban")
    return TIER_COLOURS.get(tier, "#888888")


# ---------------------------------------------------------------------------
# 1. tier_comparison.png
# ---------------------------------------------------------------------------
def plot_tier_comparison(df: pd.DataFrame, out_dir: str = "charts") -> str:
    avg = df.groupby("city")["aqi"].mean().sort_values(ascending=True)
    colours = [_city_colour(c) for c in avg.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(avg.index, avg.values, color=colours)
    ax.set_xlabel("Average AQI")
    ax.set_ylabel("City")
    ax.set_title("Average AQI by City (2019-2023)")
    patches = [mpatches.Patch(color=v, label=k) for k, v in TIER_COLOURS.items()]
    ax.legend(handles=patches, title="Tier")
    return _save(fig, os.path.join(out_dir, "tier_comparison.png"))


# ---------------------------------------------------------------------------
# 2. city_month_heatmap.png
# ---------------------------------------------------------------------------
def plot_city_month_heatmap(df: pd.DataFrame, out_dir: str = "charts") -> str:
    pivot = df.groupby(["city", df["date"].dt.month])["aqi"].mean().unstack()
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Monthly Average AQI by City")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")
    return _save(fig, os.path.join(out_dir, "city_month_heatmap.png"))


# ---------------------------------------------------------------------------
# 3. monsoon_dip.png
# ---------------------------------------------------------------------------
def plot_monsoon_dip(df: pd.DataFrame, out_dir: str = "charts") -> str:
    monthly = df.groupby(["city", df["date"].dt.month])["aqi"].mean().reset_index()
    monthly.columns = ["city", "month", "aqi"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for city in sorted(df["city"].unique()):
        sub = monthly[monthly["city"] == city]
        ax.plot(sub["month"], sub["aqi"], marker="o", color=_city_colour(city), label=city)
    ax.axvspan(7, 9, alpha=0.15, color="blue", label="Monsoon (Jul-Sep)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average AQI")
    ax.set_title("Seasonal AQI Pattern - Monsoon Effect")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    return _save(fig, os.path.join(out_dir, "monsoon_dip.png"))


# ---------------------------------------------------------------------------
# 4. yoy_trend.png
# ---------------------------------------------------------------------------
def plot_yoy_trend(df: pd.DataFrame, out_dir: str = "charts") -> str:
    yearly = df.groupby(["city", df["date"].dt.year])["aqi"].mean().reset_index()
    yearly.columns = ["city", "year", "aqi"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for city in sorted(df["city"].unique()):
        sub = yearly[yearly["city"] == city]
        ax.plot(sub["year"], sub["aqi"], marker="o", color=_city_colour(city), label=city)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Average AQI")
    ax.set_title("Year-on-Year AQI Trend")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    return _save(fig, os.path.join(out_dir, "yoy_trend.png"))


# ---------------------------------------------------------------------------
# 5. industrial_corridor.png
# ---------------------------------------------------------------------------
def plot_industrial_corridor(df: pd.DataFrame, out_dir: str = "charts") -> str:
    corridor_colours = {"Jharsuguda": "#E74C3C", "Angul": "#E67E22", "Talcher": "#922B21"}
    fig, ax = plt.subplots(figsize=(14, 5))
    for city in CORRIDOR_CITIES:
        sub = df[df["city"] == city].sort_values("date")
        colour = corridor_colours.get(city, _city_colour(city))
        ax.plot(sub["date"], sub["aqi"], alpha=0.8, color=colour, label=city, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.set_title("Western Odisha Industrial Corridor - Daily AQI")
    ax.legend()
    return _save(fig, os.path.join(out_dir, "industrial_corridor.png"))


# ---------------------------------------------------------------------------
# 6. pollutant_correlation.png
# ---------------------------------------------------------------------------
def plot_pollutant_correlation(df: pd.DataFrame, out_dir: str = "charts") -> str:
    cols = [c for c in ["pm25", "pm10", "no2", "so2", "o3", "co", "aqi"] if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Pollutant Correlation Matrix")
    ax.set_xlabel("Pollutant")
    ax.set_ylabel("Pollutant")
    return _save(fig, os.path.join(out_dir, "pollutant_correlation.png"))


# ---------------------------------------------------------------------------
# 7. diwali_spike.png
# ---------------------------------------------------------------------------
def plot_diwali_spike(df: pd.DataFrame, out_dir: str = "charts") -> str:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Filter to Oct-Nov each year
    mask = df["date"].dt.month.isin([10, 11])
    sub = df[mask].groupby("date")["aqi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub["date"], sub["aqi"], color="#555555", linewidth=0.8, label="Avg AQI (all cities)")
    for d in DIWALI_DATES:
        ts = pd.Timestamp(d)
        ax.axvline(ts, color="red", linestyle="--", linewidth=1.2,
                   label=f"Diwali {ts.year}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average AQI")
    ax.set_title("Diwali AQI Spike Pattern")
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    return _save(fig, os.path.join(out_dir, "diwali_spike.png"))


# ---------------------------------------------------------------------------
# 8. pollutant_dominance.png
# ---------------------------------------------------------------------------
def plot_pollutant_dominance(df: pd.DataFrame, out_dir: str = "charts") -> str:
    pollutants = [c for c in ["pm25", "pm10", "no2", "so2", "o3", "co"] if c in df.columns]
    avg = df.groupby("city")[pollutants].mean()
    # Sort cities by total pollution
    avg = avg.loc[avg.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, 6))
    avg.plot(kind="bar", stacked=True, ax=ax,
             colormap="tab10", edgecolor="none")
    ax.set_xlabel("City")
    ax.set_ylabel("Average Concentration")
    ax.set_title("Dominant Pollutant by City")
    ax.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, os.path.join(out_dir, "pollutant_dominance.png"))


# ---------------------------------------------------------------------------
# 9. feature_importance_comparison.png
# ---------------------------------------------------------------------------
def plot_feature_importance_comparison(
    models_dict: dict,
    feature_cols: List[str],
    out_dir: str = "charts"
) -> str:
    """
    models_dict: {city_name: fitted XGBRegressor}
    feature_cols: ordered list of feature names
    """
    cities = list(models_dict.keys())
    importance_matrix = np.zeros((len(feature_cols), len(cities)))
    for j, city in enumerate(cities):
        model = models_dict[city]
        imp = model.feature_importances_
        # Normalise 0-1
        max_imp = imp.max() if imp.max() > 0 else 1.0
        importance_matrix[:, j] = imp / max_imp
    df_imp = pd.DataFrame(importance_matrix, index=feature_cols, columns=cities)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df_imp, cmap="Blues", ax=ax, linewidths=0.3,
                vmin=0, vmax=1, annot=False)
    ax.set_title("XGBoost Feature Importance Across Cities")
    ax.set_xlabel("City")
    ax.set_ylabel("Feature")
    return _save(fig, os.path.join(out_dir, "feature_importance_comparison.png"))


# ---------------------------------------------------------------------------
# 10. feature_importance_{city}.png  (called once per city)
# ---------------------------------------------------------------------------
def plot_feature_importance_city(
    city: str,
    model: Any,
    feature_cols: List[str],
    out_dir: str = "charts"
) -> str:
    imp = model.feature_importances_
    sorted_idx = np.argsort(imp)
    colour = _city_colour(city)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        imp[sorted_idx],
        color=colour
    )
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title(f"XGBoost Feature Importance — {city}")
    filename = f"feature_importance_{city.lower()}.png"
    return _save(fig, os.path.join(out_dir, filename))


# ---------------------------------------------------------------------------
# 11. model_comparison.png
# ---------------------------------------------------------------------------
def plot_model_comparison(results: pd.DataFrame, out_dir: str = "charts") -> str:
    """results: DataFrame with columns city, model_type (lr/xgb), mae, rmse, r2"""
    lr_data  = results[results["model_type"] == "lr"].set_index("city")["mae"]
    xgb_data = results[results["model_type"] == "xgb"].set_index("city")["mae"]
    cities = sorted(results["city"].unique())
    x = np.arange(len(cities))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, [lr_data.get(c, 0) for c in cities],  width, label="Linear Regression", color="#4472C4")
    ax.bar(x + width/2, [xgb_data.get(c, 0) for c in cities], width, label="XGBoost",           color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha="right")
    ax.set_xlabel("City")
    ax.set_ylabel("MAE")
    ax.set_title("Model Comparison - MAE by City (Lower is Better)")
    ax.legend()
    return _save(fig, os.path.join(out_dir, "model_comparison.png"))


# ---------------------------------------------------------------------------
# 12. industrial_vs_urban_aqi.png
# ---------------------------------------------------------------------------
def plot_industrial_vs_urban(df: pd.DataFrame, out_dir: str = "charts") -> str:
    df = df.copy()
    df["tier"] = df["city"].map(lambda c: CITIES.get(c, {}).get("tier", "urban"))
    tier_order = ["heavy_industrial", "urban", "clean_baseline"]
    palette = {t: TIER_COLOURS[t] for t in tier_order}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="tier", y="aqi", order=tier_order,
                palette=palette, ax=ax)
    ax.set_xlabel("City Tier")
    ax.set_ylabel("AQI")
    ax.set_title("AQI Distribution by City Tier")
    return _save(fig, os.path.join(out_dir, "industrial_vs_urban_aqi.png"))
