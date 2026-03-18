import streamlit as st
import pandas as pd
import plotly.express as px
import os
from functools import reduce

st.set_page_config(page_title="Bristol-Pink Dashboard", layout="wide")

# product colours
COLOR_MAP = {
    "Americano": "#A0522D",
    "Cappuccino": "#C0A080",
    "Croissant":  "#E8A850",
}

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #e8789c !important; }
    </style>
""", unsafe_allow_html=True)

# maps each CSV file to its products
FILE_PRODUCT_MAP = {
    "Coffee_Sales.csv":    ["Americano", "Cappuccino"],
    "Croissant_Sales.csv": ["Croissant"],
}

def get_files():
    raw = [f for f in os.listdir('.') if f.endswith('.csv')]
    return {f.replace('.csv', '').replace('_', ' ').title(): f for f in raw}

file_dict = get_files()

# sidebar
st.sidebar.image("https://via.placeholder.com/150x50/161b22/ff7eb9?text=Bristol-Pink", use_container_width=True)
st.sidebar.title("Data Selection")

st.sidebar.write("**Data Sources**")
selected = []
for name in file_dict:
    if st.sidebar.checkbox(name, value=True):
        selected.append(name)

st.sidebar.divider()

st.sidebar.write("**Training Window**")
training_weeks = st.sidebar.select_slider(
    "Weeks", options=[4, 5, 6, 7, 8], value=6, label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.write("**Month**")
MONTHS = {
    "All":            None,
    "March 2025":     ("2025-03-01", "2025-03-31"),
    "April 2025":     ("2025-04-01", "2025-04-30"),
    "May 2025":       ("2025-05-01", "2025-05-31"),
    "June 2025":      ("2025-06-01", "2025-06-30"),
    "July 2025":      ("2025-07-01", "2025-07-31"),
    "August 2025":    ("2025-08-01", "2025-08-31"),
    "September 2025": ("2025-09-01", "2025-09-30"),
    "October 2025":   ("2025-10-01", "2025-10-31"),
}
selected_month = st.sidebar.radio(
    "Month", options=list(MONTHS.keys()), index=0, label_visibility="collapsed"
)

# figure out which products to show based on selected files
active_products = []
for name, filename in file_dict.items():
    if name in selected:
        active_products.extend(FILE_PRODUCT_MAP.get(filename, []))
active_products = sorted(set(active_products))

# main page
st.title("Bristol-Pink — Sales Dashboard")

if not active_products:
    st.warning("Please select a data source from the sidebar to view metrics.")
else:
    tab_analysis, tab_forecast, tab_model = st.tabs(
        ["Market Analysis", "Sales Forecast", "Model Performance"]
    )

    @st.cache_data
    def load_coffee():
        df = pd.read_csv("Coffee_Sales.csv", skiprows=1)
        df.columns = ["Date", "Cappuccino", "Americano"]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df

    @st.cache_data
    def load_croissant():
        df = pd.read_csv("Croissant_Sales.csv")
        df.columns = ["Date", "Croissant"]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df

    frames = []
    for name, filename in file_dict.items():
        if name in selected:
            if filename == "Coffee_Sales.csv":
                frames.append(load_coffee())
            elif filename == "Croissant_Sales.csv":
                frames.append(load_croissant())

    df = reduce(lambda l, r: pd.merge(l, r, on="Date", how="outer"), frames)
    df = df.sort_values("Date").reset_index(drop=True)

    active_products = [p for p in active_products if p in df.columns]

    # apply month filter
    month_range = MONTHS[selected_month]
    if month_range:
        start, end = pd.Timestamp(month_range[0]), pd.Timestamp(month_range[1])
        df = df[(df["Date"] >= start) & (df["Date"] <= end)].reset_index(drop=True)

    with tab_analysis:
        st.header("Historical Sales Overview")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Volume Mix")
            summary = df[active_products].sum().reset_index()
            summary.columns = ["Product", "Units"]
            fig_pie = px.pie(
                summary, values="Units", names="Product", hole=0.5,
                color="Product", color_discrete_map=COLOR_MAP
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Daily Trends")
            fig_line = px.line(
                df, x="Date", y=active_products, markers=True,
                color_discrete_map=COLOR_MAP
            )
            fig_line.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)

    with tab_forecast:
        st.header("Sales Forecast")
        target = st.radio("Select Product", options=active_products, horizontal=True)

        # forecast length follows the training window
        forecast_days = training_weeks * 7
        pred_dates = pd.date_range(
            start=df["Date"].max() + pd.Timedelta(days=1), periods=forecast_days
        )

        # XGBoost predictions will go here once the model is connected
        pred_df = pd.DataFrame({
            "Date": pred_dates,
            "Predicted_Sales": [None] * forecast_days
        })

        v_graph, v_table = st.tabs(["Forecast", "Data Table"])

        with v_graph:
            st.subheader(f"{forecast_days}-Day Forecast — {target}")
            fig_pred = px.line(pred_df, x="Date", y="Predicted_Sales")
            fig_pred.update_traces(line_color=COLOR_MAP.get(target), line_width=4)
            fig_pred.update_layout(template="plotly_dark", dragmode="zoom")
            st.plotly_chart(fig_pred, use_container_width=True)

        with v_table:
            st.subheader(f"{target} — Sales Data")
            st.write("Recent Sales")
            st.dataframe(df[["Date", target]].tail(10), use_container_width=True)
            st.write("Forecast")
            st.dataframe(pred_df, use_container_width=True)

    with tab_model:
        st.header("Model Performance")
        st.caption("Metrics will be populated once the XGBoost model is connected.")

        # MAE = Mean Absolute Error, lower is better
        metrics = pd.DataFrame({
            "Algorithm":    ["XGBoost"],
            "MAE":          ["—"],
            "Accuracy (%)": ["—"],
        })
        st.table(metrics)

st.markdown("---")
st.caption("Bristol-Pink Bakery © 2025")