"""
visualisations.py — Interactive Plotly chart generation for the Odisha AQI Advisor.
All functions return plotly.graph_objects.Figure objects.
"""
from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import CITIES, TIER_COLOURS, INDUSTRIAL_CITIES, CORRIDOR_CITIES, DIWALI_DATES
from src.constants import FEATURE_LABELS, TIER_LABELS, AQI_BANDS

DEFAULT_HEIGHT = 380


def _city_colour(city: str) -> str:
    tier = CITIES.get(city, {}).get("tier", "urban")
    return TIER_COLOURS.get(tier, "#888888")


def _add_aqi_bands(fig: go.Figure) -> go.Figure:
    """Add CPCB AQI band hrect annotations to a figure."""
    for lo, hi, colour, label in AQI_BANDS:
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor=colour, opacity=0.08,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=9,
        )
    return fig


# ---------------------------------------------------------------------------
# 1. Tier comparison — average AQI bar chart
# ---------------------------------------------------------------------------
def plot_tier_comparison(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    avg = df.groupby("city")["aqi"].mean().sort_values(ascending=True).reset_index()
    avg["tier"] = avg["city"].map(lambda c: CITIES.get(c, {}).get("tier", "urban"))
    avg["tier_label"] = avg["tier"].map(TIER_LABELS)
    avg["colour"] = avg["tier"].map(TIER_COLOURS)

    fig = go.Figure()
    for tier_key, tier_label in TIER_LABELS.items():
        sub = avg[avg["tier"] == tier_key]
        fig.add_trace(go.Bar(
            x=sub["aqi"], y=sub["city"],
            orientation="h",
            name=tier_label,
            marker_color=TIER_COLOURS[tier_key],
            hovertemplate="<b>%{y}</b><br>Avg AQI: %{x:.1f}<br>Tier: " + tier_label + "<extra></extra>",
        ))
    fig.update_layout(
        title="Average AQI by City (2019–2023)",
        xaxis_title="Average AQI",
        yaxis_title="City",
        height=height,
        barmode="overlay",
        legend_title="Tier",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. City-month heatmap
# ---------------------------------------------------------------------------
def plot_city_month_heatmap(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = df.groupby(["city", df["date"].dt.month])["aqi"].mean().unstack()
    pivot.columns = month_names

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names,
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        hovertemplate="City: <b>%{y}</b><br>Month: %{x}<br>Avg AQI: %{z:.0f}<extra></extra>",
        colorbar_title="AQI",
    ))
    fig.update_layout(
        title="Monthly Average AQI by City",
        xaxis_title="Month",
        yaxis_title="City",
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Monsoon dip — seasonal AQI pattern
# ---------------------------------------------------------------------------
def plot_monsoon_dip(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby(["city", df["date"].dt.month])["aqi"].mean().reset_index()
    monthly.columns = ["city", "month", "aqi"]

    fig = go.Figure()
    for city in sorted(df["city"].unique()):
        sub = monthly[monthly["city"] == city]
        fig.add_trace(go.Scatter(
            x=sub["month"], y=sub["aqi"],
            mode="lines+markers",
            name=city,
            line_color=_city_colour(city),
            hovertemplate=f"<b>{city}</b><br>Month: %{{x}}<br>Avg AQI: %{{y:.1f}}<extra></extra>",
        ))
    # Monsoon band
    fig.add_vrect(x0=6.5, x1=9.5, fillcolor="blue", opacity=0.07,
                  annotation_text="Monsoon", annotation_position="top left")
    fig.update_layout(
        title="Seasonal AQI Pattern — Monsoon Effect",
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=month_names),
        yaxis_title="Average AQI",
        height=height,
        legend_title="City",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Year-on-year trend
# ---------------------------------------------------------------------------
def plot_yoy_trend(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    yearly = df.groupby(["city", df["date"].dt.year])["aqi"].mean().reset_index()
    yearly.columns = ["city", "year", "aqi"]

    fig = go.Figure()
    for city in sorted(df["city"].unique()):
        sub = yearly[yearly["city"] == city]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["aqi"],
            mode="lines+markers",
            name=city,
            line_color=_city_colour(city),
            hovertemplate=f"<b>{city}</b><br>Year: %{{x}}<br>Annual Avg AQI: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Year-on-Year AQI Trend",
        xaxis=dict(title="Year", tickformat="d"),
        yaxis_title="Annual Average AQI",
        height=height,
        legend_title="City",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Industrial corridor — monthly averages
# ---------------------------------------------------------------------------
def plot_industrial_corridor(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    corridor_colours = {"Jharsuguda": "#E74C3C", "Angul": "#E67E22", "Talcher": "#922B21"}
    sub = df[df["city"].isin(CORRIDOR_CITIES)].copy()
    sub["month"] = sub["date"].dt.to_period("M").dt.to_timestamp()
    monthly = sub.groupby(["city", "month"])["aqi"].mean().reset_index()

    fig = go.Figure()
    for city in CORRIDOR_CITIES:
        city_data = monthly[monthly["city"] == city]
        fig.add_trace(go.Scatter(
            x=city_data["month"], y=city_data["aqi"],
            mode="lines",
            name=city,
            line_color=corridor_colours.get(city, "#888"),
            hovertemplate=f"<b>{city}</b><br>Month: %{{x|%b %Y}}<br>Avg AQI: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Western Odisha Industrial Corridor — Monthly Avg AQI",
        xaxis_title="Month",
        yaxis_title="Average AQI",
        height=height,
        legend_title="City",
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Pollutant correlation matrix
# ---------------------------------------------------------------------------
def plot_pollutant_correlation(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    cols = [c for c in ["pm25", "pm10", "no2", "so2", "aqi"] if c in df.columns]
    corr = df[cols].corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
        colorbar_title="r",
    ))
    fig.update_layout(
        title="Pollutant Correlation Matrix",
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Diwali spike
# ---------------------------------------------------------------------------
def plot_diwali_spike(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = df["date"].dt.month.isin([10, 11])
    sub = df[mask].groupby("date")["aqi"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["aqi"],
        mode="lines",
        name="Avg AQI (all cities)",
        line_color="#555555",
        hovertemplate="Date: %{x|%d %b %Y}<br>Avg AQI: %{y:.1f}<extra></extra>",
    ))
    for d in DIWALI_DATES:
        ts = pd.Timestamp(d)
        fig.add_shape(
            type="line", x0=ts, x1=ts, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", dash="dash", width=1.5),
        )
        fig.add_annotation(
            x=ts, y=1, xref="x", yref="paper",
            text=f"Diwali {ts.year}", showarrow=False,
            font=dict(size=9, color="red"), yanchor="bottom",
        )
    fig.update_layout(
        title="Diwali AQI Spike Pattern",
        xaxis_title="Date",
        yaxis_title="Average AQI",
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Pollutant dominance — stacked bar
# ---------------------------------------------------------------------------
def plot_pollutant_dominance(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    pollutants = [c for c in ["pm25", "pm10", "no2", "so2"] if c in df.columns]
    avg = df.groupby("city")[pollutants].mean()
    avg = avg.loc[avg.sum(axis=1).sort_values(ascending=False).index]

    colours = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, p in enumerate(pollutants):
        fig.add_trace(go.Bar(
            x=avg.index, y=avg[p],
            name=p.upper(),
            marker_color=colours[i % len(colours)],
            hovertemplate=f"<b>%{{x}}</b><br>{p.upper()}: %{{y:.1f}} µg/m³<extra></extra>",
        ))
    fig.update_layout(
        title="Dominant Pollutant by City",
        xaxis_title="City",
        yaxis_title="Average Concentration (µg/m³)",
        barmode="stack",
        height=height,
        legend_title="Pollutant",
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Feature importance comparison heatmap (across cities)
# ---------------------------------------------------------------------------
def plot_feature_importance_comparison(
    models_dict: dict,
    feature_cols: List[str],
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    cities = list(models_dict.keys())
    importance_matrix = np.zeros((len(feature_cols), len(cities)))
    for j, city in enumerate(cities):
        imp = models_dict[city].feature_importances_
        max_imp = imp.max() if imp.max() > 0 else 1.0
        importance_matrix[:, j] = imp / max_imp

    y_labels = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    fig = go.Figure(data=go.Heatmap(
        z=importance_matrix,
        x=cities,
        y=y_labels,
        colorscale="Blues",
        zmin=0, zmax=1,
        hovertemplate="City: <b>%{x}</b><br>Feature: <b>%{y}</b><br>Importance: %{z:.2f}<extra></extra>",
        colorbar_title="Importance",
    ))
    fig.update_layout(
        title="XGBoost Feature Importance Across Cities",
        xaxis_title="City",
        yaxis_title="Feature",
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Feature importance for a single city
# ---------------------------------------------------------------------------
def plot_feature_importance_city(
    city: str,
    model: Any,
    feature_cols: List[str],
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    imp = model.feature_importances_
    sorted_idx = np.argsort(imp)
    y_labels = [FEATURE_LABELS.get(feature_cols[i], feature_cols[i]) for i in sorted_idx]

    fig = go.Figure(go.Bar(
        x=imp[sorted_idx],
        y=y_labels,
        orientation="h",
        marker_color=_city_colour(city),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"XGBoost Feature Importance — {city}",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 11. Model comparison — MAE bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison(results: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    cities = sorted(results["city"].unique())
    lr_data  = results[results["model_type"] == "lr"].set_index("city")["mae"]
    xgb_data = results[results["model_type"] == "xgb"].set_index("city")["mae"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cities, y=[lr_data.get(c, 0) for c in cities],
        name="Linear Regression",
        marker_color="#4472C4",
        hovertemplate="<b>%{x}</b><br>LR MAE: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=cities, y=[xgb_data.get(c, 0) for c in cities],
        name="XGBoost",
        marker_color="#ED7D31",
        hovertemplate="<b>%{x}</b><br>XGB MAE: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Model Comparison — MAE by City (Lower is Better)",
        xaxis_title="City",
        yaxis_title="MAE",
        barmode="group",
        height=height,
        legend_title="Model",
    )
    return fig


# ---------------------------------------------------------------------------
# 12. Industrial vs urban AQI box plot
# ---------------------------------------------------------------------------
def plot_industrial_vs_urban(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    df = df.copy()
    df["tier"] = df["city"].map(lambda c: CITIES.get(c, {}).get("tier", "urban"))
    df["tier_label"] = df["tier"].map(TIER_LABELS)

    tier_order = ["Heavy Industrial", "Urban", "Clean Baseline"]
    fig = go.Figure()
    for tier_key, tier_label in TIER_LABELS.items():
        sub = df[df["tier"] == tier_key]["aqi"].dropna()
        fig.add_trace(go.Box(
            y=sub,
            name=tier_label,
            marker_color=TIER_COLOURS[tier_key],
            hovertemplate=f"<b>{tier_label}</b><br>AQI: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        title="AQI Distribution by City Tier",
        xaxis_title="City Tier",
        yaxis_title="AQI",
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 13. Historical AQI chart (new — replaces inline chart in app.py)
# ---------------------------------------------------------------------------
def plot_historical_aqi(
    df: pd.DataFrame,
    city: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: str = "monthly",
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """Plot historical AQI for a city with optional monthly/daily granularity.

    granularity: "monthly" (default) or "daily"
    Includes 30-day rolling average and CPCB band annotations.
    """
    city_df = df[(df["city"] == city) & (df["date"] >= start) & (df["date"] <= end)].copy()
    city_df = city_df.sort_values("date")

    fig = go.Figure()

    if granularity == "monthly":
        monthly = city_df.set_index("date")["aqi"].resample("ME").mean().reset_index()
        monthly.columns = ["date", "aqi"]
        rolling = monthly["aqi"].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Bar(
            x=monthly["date"], y=monthly["aqi"],
            name="Monthly Avg AQI",
            marker_color=_city_colour(city),
            hovertemplate="Month: %{x|%b %Y}<br>Avg AQI: %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["date"], y=rolling,
            mode="lines",
            name="3-Month Rolling Avg",
            line=dict(color="black", dash="dot"),
            hovertemplate="Month: %{x|%b %Y}<br>Rolling Avg: %{y:.1f}<extra></extra>",
        ))
    else:
        rolling = city_df["aqi"].rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=city_df["date"], y=city_df["aqi"],
            mode="lines",
            name="Daily AQI",
            line_color=_city_colour(city),
            hovertemplate="Date: %{x|%d %b %Y}<br>AQI: %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=city_df["date"], y=rolling,
            mode="lines",
            name="30-Day Rolling Avg",
            line=dict(color="black", dash="dot"),
            hovertemplate="Date: %{x|%d %b %Y}<br>30-Day Avg: %{y:.1f}<extra></extra>",
        ))

    fig = _add_aqi_bands(fig)
    fig.update_layout(
        title=f"{city} — Historical AQI ({start.date()} to {end.date()})",
        xaxis_title="Date",
        yaxis_title="AQI",
        height=height,
        legend_title="Series",
    )
    return fig


# ---------------------------------------------------------------------------
# 14. Correlation matrix (feature-level, zmin=0.5)
# ---------------------------------------------------------------------------
def plot_correlation_matrix(df: pd.DataFrame, height: int = DEFAULT_HEIGHT) -> go.Figure:
    from src.features import FEATURE_COLS
    cols = [c for c in FEATURE_COLS if c in df.columns]
    corr = df[cols].corr().round(2)
    labels = [FEATURE_LABELS.get(c, c) for c in cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmin=0.5, zmax=1,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
        colorbar_title="r",
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=height,
    )
    return fig
