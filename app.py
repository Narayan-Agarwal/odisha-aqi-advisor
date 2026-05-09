"""
app.py — Streamlit entry point for the Odisha AQI Advisor V3.
Run: streamlit run app.py  (from the odisha-aqi-advisor/ directory)
"""
import os

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st

from src.advisory import get_advisory, aqi_to_category
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

# ---------------------------------------------------------------------------
# Auto-update: fetch latest WAQI data once per session
# ---------------------------------------------------------------------------
if "data_updated" not in st.session_state:
    try:
        from src.auto_updater import run_auto_update
        with st.spinner("Fetching latest air quality data..."):
            run_auto_update()
    except Exception:
        pass  # Never crash the app if WAQI is unreachable
    st.session_state["data_updated"] = True


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

@st.cache_data(ttl=300)  # re-read featured.csv every 5 min to pick up auto-updates
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

    # Dynamic date range — always derived from actual data, never hardcoded
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
    st.sidebar.caption(
        f"Data: CPCB 2019–2023 + WAQI live updates | 10 Odisha cities\n\n"
        f"📅 Data last updated: {max_date}"
    )

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
        filtered = df[(df["city"] == city) & (df["date"] >= start) & (df["date"] <= end)]
        if len(filtered) == 0:
            st.info("📅 No data available for this date range. Please select dates within the available range shown in the sidebar.")
        else:
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
        st.info("📅 No data available for this date range. Please select dates within the available range shown in the sidebar.")
        return

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
        st.info("📅 No data available for this date range. Please select dates within the available range shown in the sidebar.")
        return

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

CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

SECTION_HEADER = (
    "<div style='background:linear-gradient(90deg,#1F3864,#2E75B6);color:white;"
    "padding:10px 18px;border-radius:8px;font-size:18px;font-weight:600;"
    "margin:24px 0 12px 0;'>{icon} {title}</div>"
)


def render_model_performance():
    import plotly.graph_objects as go_local

    st.header("📊 Model Performance")
    screen_width = _get_screen_width()
    chart_height = 250 if screen_width < 768 else 380

    try:
        results = get_model_results()
    except FileNotFoundError:
        st.error("model_results.csv not found. Run notebook 04 to train models first.")
        return

    if len(results) == 0:
        st.info("No model results available.")
        return

    # -----------------------------------------------------------------------
    # SECTION 1 — Live Model Accuracy Tracker (TOP)
    # -----------------------------------------------------------------------
    st.markdown(
        SECTION_HEADER.format(icon="📈", title="Live Model Accuracy Tracker"),
        unsafe_allow_html=True,
    )
    st.markdown("Tracks how model accuracy changes over time as new data is added and the model is retrained.")

    hist_path = "data/processed/accuracy_history.csv"
    if os.path.exists(hist_path):
        hist_df = pd.read_csv(hist_path)
        hist_df["date"] = pd.to_datetime(hist_df["date"])

        tracker_city = st.selectbox(
            "Select city for accuracy tracker",
            sorted(CITIES.keys()),
            key="tracker_city",
        )

        city_hist = hist_df[
            (hist_df["city"] == tracker_city) & (hist_df["model"] == "xgb")
        ].sort_values("date").reset_index(drop=True)

        if len(city_hist) >= 1:
            current_acc  = city_hist.iloc[-1]["category_accuracy"]
            current_mae  = city_hist.iloc[-1]["mae"]
            current_r2   = city_hist.iloc[-1]["r2"]

            if len(city_hist) >= 2:
                prev_acc  = city_hist.iloc[-2]["category_accuracy"]
                prev_mae  = city_hist.iloc[-2]["mae"]
                prev_r2   = city_hist.iloc[-2]["r2"]
                acc_delta = f"{(current_acc - prev_acc)*100:+.1f}% vs previous run"
                mae_delta = f"{current_mae - prev_mae:+.1f}"
                r2_delta  = f"{current_r2 - prev_r2:+.3f}"
            else:
                acc_delta = "First recorded run"
                mae_delta = "First run"
                r2_delta  = "First run"

            col1, col2, col3 = st.columns(3)
            col1.metric(
                label=f"Category Accuracy — {tracker_city}",
                value=f"{current_acc * 100:.1f}%",
                delta=acc_delta,
                help="Percentage of test days where model predicted the correct CPCB health category",
            )
            col2.metric(
                label="MAE",
                value=f"{current_mae:.1f} AQI units",
                delta=mae_delta if mae_delta == "First run" else mae_delta,
                delta_color="inverse",
                help="Average prediction error in AQI units. Lower is better.",
            )
            col3.metric(
                label="R²",
                value=f"{current_r2:.3f}",
                delta=r2_delta,
                help="Proportion of AQI variation explained by the model. Higher is better.",
            )

        if len(city_hist) >= 2:
            fig_hist = go_local.Figure()
            fig_hist.add_trace(go_local.Scatter(
                x=city_hist["date"],
                y=city_hist["category_accuracy"] * 100,
                mode="lines+markers",
                name="Category Accuracy %",
                line=dict(color="#2E75B6", width=2),
                marker=dict(size=8),
                hovertemplate="Date: %{x|%d %b %Y}<br>Accuracy: %{y:.1f}%<extra></extra>",
            ))
            fig_hist.update_layout(
                title=f"Model Accuracy Over Time — {tracker_city} (XGBoost)",
                xaxis_title="Date of model run",
                yaxis_title="Category Accuracy (%)",
                yaxis=dict(range=[0, 100]),
                height=350,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Each point represents one model training run. As more data is added and the model is retrained, accuracy is tracked here automatically.")
        else:
            st.info("Retrain the model at least twice to see the accuracy trend chart.")
    else:
        st.info("accuracy_history.csv not found. Re-run notebook 04 to generate it.")

    # -----------------------------------------------------------------------
    # SECTION 2 — Model Comparison Matrix
    # -----------------------------------------------------------------------
    st.markdown(
        SECTION_HEADER.format(icon="⚖️", title="Model Comparison — XGBoost vs Linear Regression"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "This table compares how accurately each model predicted AQI for every city. "
        "**Lower MAE and RMSE = more accurate. Higher R² = better fit.** "
        "XGBoost (orange) is the primary model. Linear Regression (blue) is the baseline. "
        "Green cells highlight the better-performing model for each metric."
    )

    xgb_df = results[results["model_type"] == "xgb"].set_index("city")
    lr_df  = results[results["model_type"] == "lr"].set_index("city")

    has_acc = "category_accuracy" in results.columns
    compare_data = {
        "XGBoost MAE":     xgb_df["mae"],
        "Linear Reg MAE":  lr_df["mae"],
        "XGBoost RMSE":    xgb_df["rmse"],
        "Linear Reg RMSE": lr_df["rmse"],
        "XGBoost R²":      xgb_df["r2"],
        "Linear Reg R²":   lr_df["r2"],
    }
    if has_acc:
        compare_data["XGBoost Cat. Acc."]    = xgb_df["category_accuracy"]
        compare_data["Linear Reg Cat. Acc."] = lr_df["category_accuracy"]

    compare_df = pd.DataFrame(compare_data).round(3)

    def highlight_better(row):
        styles = [""] * len(row)
        cols = list(row.index)
        pairs = [
            ("XGBoost MAE",  "Linear Reg MAE",  "lower"),
            ("XGBoost RMSE", "Linear Reg RMSE", "lower"),
            ("XGBoost R²",   "Linear Reg R²",   "higher"),
        ]
        if has_acc:
            pairs.append(("XGBoost Cat. Acc.", "Linear Reg Cat. Acc.", "higher"))
        for xgb_col, lr_col, better in pairs:
            if xgb_col not in cols or lr_col not in cols:
                continue
            xi, li = cols.index(xgb_col), cols.index(lr_col)
            if better == "lower":
                winner = xi if row[xgb_col] <= row[lr_col] else li
            else:
                winner = xi if row[xgb_col] >= row[lr_col] else li
            styles[winner] = "background-color:#D5F5E3;font-weight:bold"
        return styles

    st.dataframe(
        compare_df.style.apply(highlight_better, axis=1),
        use_container_width=True,
    )

    # Grouped bar chart — MAE comparison
    cities_list = compare_df.index.tolist()
    fig_compare = go_local.Figure()
    fig_compare.add_trace(go_local.Bar(
        name="XGBoost MAE",
        x=cities_list,
        y=compare_df["XGBoost MAE"].tolist(),
        marker_color="#E67E22",
        text=compare_df["XGBoost MAE"].round(1).tolist(),
        textposition="outside",
        hovertemplate="XGBoost<br>City: %{x}<br>MAE: %{y:.1f} AQI units<extra></extra>",
    ))
    fig_compare.add_trace(go_local.Bar(
        name="Linear Regression MAE",
        x=cities_list,
        y=compare_df["Linear Reg MAE"].tolist(),
        marker_color="#2E75B6",
        text=compare_df["Linear Reg MAE"].round(1).tolist(),
        textposition="outside",
        hovertemplate="Linear Regression<br>City: %{x}<br>MAE: %{y:.1f} AQI units<extra></extra>",
    ))
    fig_compare.update_layout(
        barmode="group",
        title="MAE Comparison — XGBoost vs Linear Regression (lower = more accurate)",
        xaxis_title="City",
        yaxis_title="MAE (AQI units)",
        height=420,
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.caption("Lower bar = more accurate model. Green highlighted cells in the table above show which model won for each city and metric.")

    # YoY trend
    try:
        df_feat = get_featured()
        fig_yoy = plot_yoy_trend(df_feat, height=chart_height)
        st.plotly_chart(fig_yoy, use_container_width=True)
        st.caption("Year-on-year average AQI trend across all cities.")
    except FileNotFoundError:
        st.info("featured.csv not found — YoY chart unavailable.")

    # -----------------------------------------------------------------------
    # SECTION 3 — Confusion Matrix (BOTTOM)
    # -----------------------------------------------------------------------
    st.markdown(
        SECTION_HEADER.format(icon="🔲", title="Prediction Confusion Matrix — Category Accuracy"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "A confusion matrix shows how often the model predicted the correct CPCB air quality category. "
        "Each **row** is the **actual category** on that day. "
        "Each **column** is the **predicted category** the model gave. "
        "Numbers on the diagonal (top-left to bottom-right) are correct predictions. "
        "Numbers off the diagonal are mistakes — for example, the model predicted Moderate "
        "but the actual category was Poor."
    )

    cm_city = st.selectbox("Select city for confusion matrix", sorted(CITIES.keys()), key="cm_city")

    cm_path = f"data/processed/confusion_matrix_{cm_city.lower()}.csv"
    if os.path.exists(cm_path):
        cm_df = pd.read_csv(cm_path, index_col=0)
        cm_df = cm_df.reindex(index=CATS, columns=CATS, fill_value=0)

        row_sums = cm_df.sum(axis=1).replace(0, 1)
        cm_norm = cm_df.div(row_sums, axis=0).round(2)

        fig_cm = ff.create_annotated_heatmap(
            z=cm_norm.values.tolist(),
            x=CATS,
            y=CATS,
            annotation_text=cm_df.values.astype(int).astype(str).tolist(),
            colorscale="Blues",
            showscale=True,
        )
        fig_cm.update_layout(
            title=f"Confusion Matrix — {cm_city} (actual vs predicted CPCB category)",
            xaxis_title="Predicted Category",
            yaxis_title="Actual Category",
            xaxis=dict(side="bottom"),
            height=500,
        )
        fig_cm.update_xaxes(tickangle=30)
        st.plotly_chart(fig_cm, use_container_width=True)

        total   = cm_df.values.sum()
        correct = sum(cm_df.iloc[i, i] for i in range(len(CATS)))
        accuracy = correct / total if total > 0 else 0.0
        st.metric(
            "Category Prediction Accuracy",
            f"{accuracy * 100:.1f}%",
            help="Percentage of days where the model predicted the correct CPCB health category",
        )
    else:
        st.info(f"Confusion matrix not yet generated for {cm_city}. Re-run notebook 04.")

    with st.expander("📖 What do these terms mean?"):
        st.markdown(
            "| Term | Meaning |\n|---|---|\n"
            "| **Diagonal cells (blue)** | Correct predictions — model predicted the right CPCB category |\n"
            "| **Off-diagonal cells** | Wrong predictions — model predicted a different category than actual |\n"
            "| **True Positive (TP)** | Model correctly predicted a specific category (diagonal value for that category) |\n"
            "| **False Positive (FP)** | Model predicted this category but actual was different (column sum minus TP) |\n"
            "| **False Negative (FN)** | Actual was this category but model predicted something else (row sum minus TP) |\n"
            "| **Precision** | Of all days the model said were Poor, what fraction actually were Poor |\n"
            "| **Recall** | Of all actual Poor days, what fraction did the model correctly identify |\n"
            "| **Misclassification** | Most common error is predicting one category off — e.g. Moderate instead of Poor. "
            "This is acceptable because adjacent categories have very close AQI values |"
        )
        st.markdown(
            "**Important note:** Minor misclassifications between adjacent categories "
            "(e.g. Moderate vs Poor) are expected because the AQI boundary is a single "
            "number (200). A prediction of 198 vs actual 202 is only 4 AQI units apart "
            "but crosses a category boundary — this appears as an error in the confusion "
            "matrix even though it is practically very close."
        )


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
