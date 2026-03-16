"""
app.py — Streamlit entry point for the Odisha AQI Advisor.
Run: streamlit run app.py  (from the odisha-aqi-advisor/ directory)
"""
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from src.advisory import get_advisory
from src.data_loader import (
    CITIES, TIER_COLOURS, INDUSTRIAL_CITIES, CORRIDOR_CITIES,
    load_featured_csv, load_model_results, load_model, load_feature_columns,
)
from src.features import FEATURE_COLS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Odisha AQI Advisor",
    page_icon="🌫️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def get_featured() -> pd.DataFrame:
    return load_featured_csv()


@st.cache_data
def get_model_results() -> pd.DataFrame:
    return load_model_results()


@st.cache_data
def get_feature_columns() -> list:
    return load_feature_columns()


@st.cache_resource
def get_model(city: str, model_type: str):
    return load_model(city, model_type)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def build_sidebar(df: pd.DataFrame):
    st.sidebar.title("🌫️ Odisha AQI Advisor")
    st.sidebar.markdown("---")

    # City selector grouped by tier
    tier_groups = {
        "🏭 Heavy Industrial": [c for c, v in CITIES.items() if v["tier"] == "heavy_industrial"],
        "🏙️ Urban":            [c for c, v in CITIES.items() if v["tier"] == "urban"],
        "🌿 Clean Baseline":   [c for c, v in CITIES.items() if v["tier"] == "clean_baseline"],
    }
    all_cities = [c for cities in tier_groups.values() for c in cities]

    city_options = []
    for group, cities in tier_groups.items():
        city_options.append(f"── {group} ──")
        city_options.extend(cities)

    # Default to first real city
    default_idx = city_options.index(all_cities[0])
    selected_raw = st.sidebar.selectbox("Select City", city_options, index=default_idx)
    city = selected_raw if selected_raw in all_cities else all_cities[0]

    # Date range
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: CPCB 2019–2023 | 10 Odisha cities")

    return city, pd.Timestamp(start_date), pd.Timestamp(end_date)


# ---------------------------------------------------------------------------
# Tab 1: City Dashboard
# ---------------------------------------------------------------------------

def render_city_dashboard(df: pd.DataFrame, city: str, start: pd.Timestamp, end: pd.Timestamp):
    st.header(f"📍 {city} — City Dashboard")

    # Load model (XGBoost preferred, fallback to LR)
    model_type_used = "xgb"
    try:
        model = get_model(city, "xgb")
    except FileNotFoundError:
        st.warning("XGBoost model not found — falling back to Linear Regression.")
        model_type_used = "lr"
        try:
            model = get_model(city, "lr")
        except FileNotFoundError:
            st.error("No model found for this city. Run notebook 04 first.")
            return

    # Build feature row from latest available data
    city_df = df[df["city"] == city].sort_values("date")
    feat_cols = get_feature_columns()
    latest = city_df.dropna(subset=feat_cols).iloc[-1]
    X_latest = latest[feat_cols].values.reshape(1, -1)
    pred_aqi = float(model.predict(X_latest)[0])

    # Advisory
    category, message, colour = get_advisory(pred_aqi)

    # Prediction card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Next-Day AQI Forecast", f"{pred_aqi:.0f}", help=f"Model: {model_type_used.upper()}")
    with col2:
        st.markdown(
            f"<div style='background:{colour};padding:12px;border-radius:8px;"
            f"color:white;font-weight:bold;text-align:center'>{category}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.info(message)

    st.markdown("---")

    # Historical AQI line chart with band shading
    filtered = city_df[(city_df["date"] >= start) & (city_df["date"] <= end)]
    if filtered.empty:
        st.warning("No data for selected date range.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(filtered["date"], filtered["aqi"], color=TIER_COLOURS.get(CITIES[city]["tier"], "#888"), linewidth=0.9)
        # AQI band shading
        bands = [(0, 50, "#00B050", "Good"), (51, 100, "#92D050", "Satisfactory"),
                 (101, 200, "#FFFF00", "Moderate"), (201, 300, "#FF7C00", "Poor"),
                 (301, 400, "#FF0000", "Very Poor"), (401, 500, "#7B0023", "Severe")]
        for lo, hi, col, lbl in bands:
            ax.axhspan(lo, hi, alpha=0.08, color=col)
        ax.set_xlabel("Date"); ax.set_ylabel("AQI")
        ax.set_title(f"{city} — Historical AQI ({start.date()} to {end.date()})")
        st.pyplot(fig)
        plt.close(fig)

    # Feature importance chart
    fi_path = os.path.join("charts", f"feature_importance_{city.lower()}.png")
    if os.path.exists(fi_path):
        st.subheader("Feature Importance (XGBoost)")
        st.image(fi_path)
    else:
        st.info("Feature importance chart not found. Run notebook 04 first.")


# ---------------------------------------------------------------------------
# Tab 2: Compare Cities
# ---------------------------------------------------------------------------

def render_compare_cities(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    st.header("🏙️ Compare Cities")

    filtered = df[(df["date"] >= start) & (df["date"] <= end)]

    # Static charts
    for fname, caption in [
        ("tier_comparison.png",           "Average AQI by City Tier"),
        ("industrial_vs_urban_aqi.png",   "AQI Distribution by Tier"),
        ("city_month_heatmap.png",        "Monthly AQI Heatmap"),
        ("pollutant_dominance.png",       "Dominant Pollutant by City"),
        ("feature_importance_comparison.png", "Feature Importance Across Cities"),
    ]:
        path = os.path.join("charts", fname)
        if os.path.exists(path):
            st.subheader(caption)
            st.image(path)

    st.markdown("---")

    if filtered.empty:
        st.warning("No data for selected date range.")
        return

    # Live multi-line AQI chart
    st.subheader("Multi-City AQI — Selected Date Range")
    fig, ax = plt.subplots(figsize=(14, 5))
    for city in sorted(filtered["city"].unique()):
        sub = filtered[filtered["city"] == city].sort_values("date")
        tier = CITIES.get(city, {}).get("tier", "urban")
        ax.plot(sub["date"], sub["aqi"], label=city, color=TIER_COLOURS.get(tier, "#888"), alpha=0.8, linewidth=0.9)
    ax.set_xlabel("Date"); ax.set_ylabel("AQI")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    st.pyplot(fig); plt.close(fig)

    # Mean AQI bar chart
    st.subheader("Mean AQI per City — Selected Date Range")
    means = filtered.groupby("city")["aqi"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    colours = [TIER_COLOURS.get(CITIES.get(c, {}).get("tier", "urban"), "#888") for c in means.index]
    ax2.bar(means.index, means.values, color=colours)
    ax2.set_xlabel("City"); ax2.set_ylabel("Mean AQI")
    ax2.tick_params(axis="x", rotation=45)
    patches = [mpatches.Patch(color=v, label=k) for k, v in TIER_COLOURS.items()]
    ax2.legend(handles=patches)
    st.pyplot(fig2); plt.close(fig2)


# ---------------------------------------------------------------------------
# Tab 3: Industrial Corridor
# ---------------------------------------------------------------------------

def render_industrial_corridor(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    st.header("🏭 Industrial Corridor")

    industrial_df = df[df["city"].isin(INDUSTRIAL_CITIES)]
    filtered = industrial_df[(industrial_df["date"] >= start) & (industrial_df["date"] <= end)]

    # Live corridor chart
    st.subheader("Corridor Cities — Daily AQI")
    corridor_filtered = filtered[filtered["city"].isin(CORRIDOR_CITIES)]
    if corridor_filtered.empty:
        st.warning("No corridor data for selected range.")
    else:
        corridor_colours = {"Jharsuguda": "#E74C3C", "Angul": "#E67E22", "Talcher": "#922B21"}
        fig, ax = plt.subplots(figsize=(14, 4))
        for city in CORRIDOR_CITIES:
            sub = corridor_filtered[corridor_filtered["city"] == city].sort_values("date")
            ax.plot(sub["date"], sub["aqi"], label=city, color=corridor_colours.get(city, "#888"), linewidth=0.9)
        ax.set_xlabel("Date"); ax.set_ylabel("AQI")
        ax.legend()
        st.pyplot(fig); plt.close(fig)

    # Static charts
    for fname, caption in [
        ("diwali_spike.png",         "Diwali AQI Spike Pattern"),
        ("pollutant_correlation.png","Pollutant Correlation Matrix"),
        ("monsoon_dip.png",          "Monsoon Effect on AQI"),
    ]:
        path = os.path.join("charts", fname)
        if os.path.exists(path):
            st.subheader(caption)
            st.image(path)


# ---------------------------------------------------------------------------
# Tab 4: Model Performance
# ---------------------------------------------------------------------------

def render_model_performance():
    st.header("📊 Model Performance")

    try:
        results = get_model_results()
    except FileNotFoundError:
        st.error("model_results.csv not found. Run notebook 04 to train models first.")
        st.code("jupyter nbconvert --to notebook --execute notebooks/04_modelling.ipynb")
        return

    st.subheader("Results Table")
    st.dataframe(results[["city", "model_type", "rmse", "mae", "r2"]].round(3), use_container_width=True)

    for fname, caption in [
        ("model_comparison.png", "MAE Comparison: Linear Regression vs XGBoost"),
        ("yoy_trend.png",        "Year-on-Year AQI Trend"),
    ]:
        path = os.path.join("charts", fname)
        if os.path.exists(path):
            st.subheader(caption)
            st.image(path)
        else:
            st.info(f"{fname} not found. Run notebook 03/04 first.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        df = get_featured()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    city, start, end = build_sidebar(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 City Dashboard",
        "🏙️ Compare Cities",
        "🏭 Industrial Corridor",
        "📊 Model Performance",
    ])

    with tab1:
        render_city_dashboard(df, city, start, end)
    with tab2:
        render_compare_cities(df, start, end)
    with tab3:
        render_industrial_corridor(df, start, end)
    with tab4:
        render_model_performance()


if __name__ == "__main__":
    main()
