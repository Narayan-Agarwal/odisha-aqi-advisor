"""
app.py — Streamlit entry point for the Odisha AQI Advisor V3.
Run: streamlit run app.py  (from the odisha-aqi-advisor/ directory)
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.advisory import get_advisory
from src.constants import AQI_BANDS, TIER_LABELS
from src.data_loader import (
    CITIES, TIER_COLOURS, INDUSTRIAL_CITIES, CORRIDOR_CITIES,
    load_featured_csv, load_model_results, load_model, load_feature_columns,
)
from src.features import FEATURE_COLS
from src.visualisations import (
    plot_tier_comparison, plot_city_month_heatmap, plot_monsoon_dip,
    plot_yoy_trend, plot_industrial_corridor, plot_pollutant_correlation,
    plot_diwali_spike, plot_pollutant_dominance, plot_feature_importance_comparison,
    plot_feature_importance_city, plot_model_comparison, plot_industrial_vs_urban,
    plot_historical_aqi,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Odisha AQI Advisor",
    page_icon="🌫️",
    layout="wide",
)

# Fix sidebar width to 260px
st.markdown(
    "<style>[data-testid='stSidebar'] { min-width: 260px; max-width: 260px; }</style>",
    unsafe_allow_html=True,
)

# JS snippet to inject screen_width as query param
st.components.v1.html(
    """
    <script>
    const w = window.innerWidth;
    const url = new URL(window.location.href);
    if (url.searchParams.get('screen_width') !== String(w)) {
        url.searchParams.set('screen_width', w);
        window.history.replaceState({}, '', url.toString());
        window.location.reload();
    }
    </script>
    """,
    height=0,
)


def layout_columns(screen_width: int) -> int:
    """Returns 1 if screen_width < 768, else 2."""
    return 1 if screen_width < 768 else 2


def _get_screen_width() -> int:
    try:
        return int(st.query_params.get("screen_width", 1024))
    except (ValueError, TypeError):
        return 1024


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

    # Dark mode toggle
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
    if dark_mode:
        st.markdown(
            "<style>:root { color-scheme: dark; } "
            ".stApp { background-color: #0e1117; color: #fafafa; }</style>",
            unsafe_allow_html=True,
        )

    # Plain alphabetical city list
    all_cities = sorted(CITIES.keys())
    city = st.sidebar.selectbox("Select City", all_cities)

    # Dynamic date range — always derived from actual data
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_start = max(min_date, (df["date"].max() - pd.Timedelta(days=365)).date())
    default_end = max_date

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, default_end

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data: CPCB 2019–2023 + WAQI live updates | 10 Odisha cities\n\nRange: {min_date} → {max_date}")

    return city, pd.Timestamp(start_date), pd.Timestamp(end_date)


# ---------------------------------------------------------------------------
# Tab 1: City Dashboard
# ---------------------------------------------------------------------------

def render_city_dashboard(df: pd.DataFrame, city: str, start: pd.Timestamp, end: pd.Timestamp):
    screen_width = _get_screen_width()
    chart_height = 250 if screen_width < 768 else 380

    # Welcome banner
    st.info("👋 Welcome to the Odisha AQI Advisor — explore air quality forecasts and trends for 10 cities across Odisha.")

    # Load model
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

    category, message, colour = get_advisory(pred_aqi)

    # Prediction card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Next-Day AQI Forecast", f"{pred_aqi:.0f}", help=f"Model: {model_type_used.upper()}")
    with col2:
        # Fixed-width badge so width doesn't shift between categories
        st.markdown(
            f"<div style='background:{colour};padding:12px;border-radius:8px;"
            f"color:white;font-weight:bold;text-align:center;min-width:160px'>{category}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.info(message)

    # What is AQI? expander
    with st.expander("ℹ️ What is AQI?"):
        st.markdown(
            "The **Air Quality Index (AQI)** is a number used by government agencies to communicate "
            "how polluted the air currently is or how polluted it is forecast to become. "
            "It is calculated from concentrations of PM2.5, PM10, SO₂, and NO₂ using the CPCB formula. "
            "A lower AQI means cleaner air.\n\n"
            "| Range | Category |\n|---|---|\n"
            "| 0–50 | Good |\n| 51–100 | Satisfactory |\n| 101–200 | Moderate |\n"
            "| 201–300 | Poor |\n| 301–400 | Very Poor |\n| 401–500 | Severe |"
        )

    st.markdown("---")

    # Historical AQI + Feature importance side by side
    ncols = layout_columns(screen_width)
    if ncols == 2:
        left, right = st.columns(2)
    else:
        left = right = st.container()

    with left:
        granularity = st.radio("Chart granularity", ["Monthly", "Daily"], horizontal=True, key="gran")
        fig_hist = plot_historical_aqi(
            df, city, start, end,
            granularity=granularity.lower(),
            height=chart_height,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(f"Historical AQI for {city} from {start.date()} to {end.date()}.")

    with right:
        try:
            xgb_model = get_model(city, "xgb")
            fig_fi = plot_feature_importance_city(city, xgb_model, feat_cols, height=chart_height)
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("XGBoost feature importance — which inputs drive the forecast most.")
        except FileNotFoundError:
            st.info("Feature importance unavailable — XGBoost model not found.")


# ---------------------------------------------------------------------------
# Tab 2: Compare Cities
# ---------------------------------------------------------------------------

def render_compare_cities(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    st.header("🏙️ Compare Cities")
    screen_width = _get_screen_width()
    chart_height = 250 if screen_width < 768 else 380

    filtered = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(filtered) == 0:
        st.info("No data available for the selected date range. Please adjust the date filter.")
        st.stop()

    # Tier comparison + box plot side by side
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_tier_comparison(filtered, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Average AQI per city, coloured by tier.")
    with col2:
        fig = plot_industrial_vs_urban(filtered, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AQI distribution across city tiers.")

    # Full-width heatmap
    fig = plot_city_month_heatmap(filtered, height=chart_height)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Monthly average AQI heatmap — darker = worse air quality.")

    # Pollutant dominance + monsoon side by side
    col3, col4 = st.columns(2)
    with col3:
        fig = plot_pollutant_dominance(filtered, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Stacked average pollutant concentrations per city.")
    with col4:
        fig = plot_monsoon_dip(filtered, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Seasonal AQI pattern — note the monsoon dip in Jul–Sep.")

    # Full-width feature importance comparison
    try:
        xgb_models = {}
        feat_cols = get_feature_columns()
        for city in CITIES:
            try:
                xgb_models[city] = get_model(city, "xgb")
            except FileNotFoundError:
                pass
        if xgb_models:
            fig = plot_feature_importance_comparison(xgb_models, feat_cols, height=chart_height)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("XGBoost feature importance heatmap across all cities.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tab 3: Industrial Corridor
# ---------------------------------------------------------------------------

def render_industrial_corridor(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    st.header("🏭 Industrial Corridor")
    screen_width = _get_screen_width()
    chart_height = 250 if screen_width < 768 else 380

    st.markdown(
        "The western Odisha industrial corridor — Jharsuguda, Angul, and Talcher — hosts "
        "some of India's largest aluminium smelters, coal-fired power plants, and coalfields. "
        "This tab explores how industrial activity drives AQI patterns in the region."
    )

    filtered = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(filtered) == 0:
        st.info("No data available for the selected date range. Please adjust the date filter.")
        st.stop()

    # Full-width corridor chart
    fig = plot_industrial_corridor(filtered, height=chart_height)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Monthly average AQI for the three corridor cities.")

    # Diwali spike + pollutant correlation side by side
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_diwali_spike(filtered, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AQI spikes around Diwali (Oct–Nov) across all cities.")
    with col2:
        fig = plot_pollutant_correlation(filtered, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation between pollutants — PM2.5 and PM10 tend to move together.")


# ---------------------------------------------------------------------------
# Tab 4: Model Performance
# ---------------------------------------------------------------------------

def render_model_performance():
    st.header("📊 Model Performance")
    screen_width = _get_screen_width()
    chart_height = 250 if screen_width < 768 else 380

    st.markdown(
        "Two models are trained per city: **Linear Regression** (fast, interpretable baseline) "
        "and **XGBoost** (gradient-boosted trees, typically more accurate). "
        "MAE (Mean Absolute Error) measures average prediction error in AQI units — lower is better. "
        "R² measures how much variance the model explains — closer to 1.0 is better."
    )

    try:
        results = get_model_results()
    except FileNotFoundError:
        st.error("model_results.csv not found. Run notebook 04 to train models first.")
        return

    if len(results) == 0:
        st.info("No model results available.")
        st.stop()

    # Rename model_type for display
    display = results.copy()
    display["model_type"] = display["model_type"].map({"lr": "Linear Regression", "xgb": "XGBoost"})
    st.dataframe(
        display[["city", "model_type", "rmse", "mae", "r2"]].round(3),
        use_container_width=True,
    )

    # MAE comparison + YoY trend side by side
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_model_comparison(results, height=chart_height)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("MAE comparison between Linear Regression and XGBoost per city.")
    with col2:
        try:
            df = get_featured()
            fig = plot_yoy_trend(df, height=chart_height)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Year-on-year average AQI trend across all cities.")
        except FileNotFoundError:
            st.info("featured.csv not found — YoY chart unavailable.")


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
