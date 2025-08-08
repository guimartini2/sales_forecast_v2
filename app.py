"""
Amazon Replenishment Forecast Streamlit App (Amazon Branded)

Key improvements/fixes in this build:
- Robust CSV loading for both Sales and Amazon Sell-Out Forecast (auto-detect headers, separators, skiprows)
- Proper handling of UploadedFile (rewind before each read attempt)
- Plotly dual-axis fixed (primary='y', secondary='y2')
- Page icon fixed (emoji, not remote URL)
- Consistent 'W-MON' weekly alignment across history, forecast, and overrides
- Projection Type toggle now affects which metric is on the primary axis
- Safer Weeks-of-Cover logic (targets based on recent average weekly demand)
- Graceful fallbacks when Plotly/Prophet/ARIMA/XGBoost are missing
"""

import os
import re
from io import BytesIO
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Optional plotting/forecasting libraries
# ----------------------------
PLOTLY_INSTALLED = False
try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    pass

PROPHET_INSTALLED = False
try:
    from prophet import Prophet
    PROPHET_INSTALLED = True
except ImportError:
    try:
        from fbprophet import Prophet  # fallback for older envs
        PROPHET_INSTALLED = True
    except ImportError:
        pass

ARIMA_INSTALLED = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_INSTALLED = True
except ImportError:
    pass

XGB_INSTALLED = False
try:
    import xgboost as xgb  # noqa: F401
    XGB_INSTALLED = True
except ImportError:
    pass

# ----------------------------
# Amazon branding
# ----------------------------
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# ----------------------------
# Page setup (icon must be emoji or local bytes)
# ----------------------------
st.set_page_config(page_title="Amazon Replenishment Forecast", page_icon="🛒", layout="wide")
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>",
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
fcst_file = st.sidebar.file_uploader("Amazon Sell-Out Forecast CSV", type=["csv"])

projection_type = st.sidebar.selectbox("Projection Type (primary Y-axis)", ["Units", "Sales $"])
init_inv = st.sidebar.number_input("Current On-Hand Inventory (units)", min_value=0, value=26730, step=1)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=10.0, step=0.01)

# Model selection
model_opts = []
if PROPHET_INSTALLED:
    model_opts.append("Prophet")
if ARIMA_INSTALLED:
    model_opts.append("ARIMA")
if XGB_INSTALLED:
    model_opts.append("XGBoost (naive last value)")  # placeholder behavior below
if not model_opts:
    st.error("Install at least one forecasting engine: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Forecast Model", model_opts)

# Parameters
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = int(st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12))

st.markdown("---")
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to generate the forecast and replenishment plan.")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
def _maybe_seek_start(obj):
    """Rewind a file-like object if it supports seek()."""
    if hasattr(obj, "seek"):
        try:
            obj.seek(0)
        except Exception:
            pass

def read_sales_csv(src):
    """
    Robust reader for the Sales history CSV.
    - Tries skiprows 0/1
    - Detects dates like 'Start - End' or direct date.
    - Returns a DataFrame with columns: ['Week_Start', <metric cols>...]
    """
    # If path-like string provided and exists, pass it through; if UploadedFile, rewind before every attempt
    attempts = []
    for skip in (0, 1):
        try:
            _maybe_seek_start(src)
            t = pd.read_csv(src, skiprows=skip, engine="python")
            if t.empty or t.shape[1] < 1:
                continue

            first_col = t.columns[0]
            # Try "Start - End" format or direct date
            starts = (
                t[first_col].astype(str).str.split(" - ").str[0].str.strip()
                if t[first_col].astype(str).str.contains(" - ").any()
                else t[first_col].astype(str)
            )
            dates = pd.to_datetime(starts, errors="coerce")
            t["Week_Start"] = dates
            t = t.dropna(subset=["Week_Start"])
            if t["Week_Start"].notna().sum() >= 4:  # need enough points
                return t
        except Exception as e:
            attempts.append((skip, str(e)))

    st.error("Could not parse Sales history CSV; please verify the header and first column are dates (or 'Start - End').")
    st.stop()

def try_parse_any_date(text):
    """
    Try multiple date formats commonly seen in Amazon exports.
    Returns pd.Timestamp or NaT.
    """
    s = str(text).strip()
    # Attempt explicit formats first
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d):
            return d
    # Fallback: let pandas guess
    return pd.to_datetime(s, errors="coerce")

def read_amazon_forecast_csv(src):
    """
    Robust reader for Amazon Sell-Out Forecast (wide headers containing dates).
    - Tries multiple separators and skiprows
    - Parses dates from parentheses if present, else from full header
    - Aggregates duplicate week columns
    Returns DataFrame with columns: ['Week_Start', 'Amazon_Sellout_Forecast']
    or an empty DataFrame if cannot parse.
    """
    df_raw = None
    for skip in (0, 1, 2):
        for sep in (",", ";", "\t", "|"):
            try:
                _maybe_seek_start(src)
                tmp = pd.read_csv(src, sep=sep, skiprows=skip, engine="python")
                if not tmp.empty and tmp.shape[1] > 1:
                    df_raw = tmp
                    break
            except Exception:
                continue
        if df_raw is not None:
            break

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"])

    # Use first row (assume one SKU per file; adapt as needed)
    row0 = df_raw.iloc[0]
    rec = []
    year_now = datetime.now().year

    for col in df_raw.columns:
        col_str = str(col)
        # Prefer content inside parentheses, e.g., "(12 Jan)" or "(12 January)"
        if "(" in col_str and ")" in col_str:
            inside = col_str.split("(")[-1].split(")")[0].strip()
            # Try with current year appended, then raw
            dt = try_parse_any_date(f"{inside} {year_now}")
            if pd.isna(dt):
                dt = try_parse_any_date(inside)
        else:
            dt = try_parse_any_date(col_str)

        if pd.notna(dt):
            val = pd.to_numeric(str(row0[col]).replace(",", ""), errors="coerce")
            if pd.notna(val):
                rec.append({"Week_Start": dt, "Amazon_Sellout_Forecast": int(round(val))})

    if not rec:
        return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"])

    df_up = pd.DataFrame(rec).dropna(subset=["Week_Start"])
    # Align to Monday week start
    df_up["Week_Start"] = pd.to_datetime(df_up["Week_Start"]).dt.to_period("W-MON").start_time
    # If multiple headers map to the same week, sum them
    df_up = df_up.groupby("Week_Start", as_index=False)["Amazon_Sellout_Forecast"].sum().sort_values("Week_Start")
    return df_up

def future_weeks(start_date, periods):
    """Return a DatetimeIndex of Mondays for the requested number of weeks starting AFTER start_date."""
    # Next Monday after start_date
    start_fc = (start_date + pd.offsets.Week(weekday=0))  # 0 = Monday
    if start_fc <= start_date:
        start_fc = start_fc + pd.offsets.Week(weekday=0)
    return pd.date_range(start=start_fc, periods=periods, freq="W-MON")

# ----------------------------
# Load data paths (default fallbacks only if the files actually exist)
# ----------------------------
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_up = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

sales_path = sales_file if sales_file is not None else (default_sales if isinstance(default_sales, str) and os.path.exists(default_sales) else None)
up_path = fcst_file if fcst_file is not None else (default_up if isinstance(default_up, str) and os.path.exists(default_up) else None)

if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

# ----------------------------
# Load & clean sales history
# ----------------------------
df_raw = read_sales_csv(sales_path)

# Determine metric column (prefer units/qty)
candidate = next((c for c in df_raw.columns if re.search(r"(unit|qty|quantity)", str(c), re.IGNORECASE)), None)
if candidate is None:
    # fallback: use the first numeric column after Week_Start
    numeric_cols = [c for c in df_raw.columns if c != "Week_Start" and pd.api.types.is_numeric_dtype(df_raw[c])]
    candidate = numeric_cols[0] if numeric_cols else df_raw.columns[1]

y_col = candidate
forecast_label = "Forecast_Units"
y_label = "Units"

# Historical series (coerce numeric)
df_hist = pd.DataFrame({
    "Week_Start": pd.to_datetime(df_raw["Week_Start"]).dt.to_period("W-MON").start_time,
    "y": pd.to_numeric(
        df_raw[y_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    ).fillna(0),
})
# Keep only up to current week
today = pd.to_datetime(datetime.now().date())
df_hist = df_hist[df_hist["Week_Start"] <= today]
df_hist = df_hist.groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")  # ensure weekly and sorted

if df_hist.empty or df_hist["y"].sum() == 0:
    st.error("No valid historical data (dates or units).")
    st.stop()

# ----------------------------
# Forecast dates (W-MON)
# ----------------------------
last_hist_week = df_hist["Week_Start"].max()
future_idx = future_weeks(max(last_hist_week, today), periods)

# ----------------------------
# Generate forecast
# ----------------------------
if model_choice == "Prophet":
    # Prophet expects columns 'ds' and 'y'
    m = Prophet(weekly_seasonality=True)
    m.fit(df_hist.rename(columns={"Week_Start": "ds", "y": "y"}))
    fut = pd.DataFrame({"ds": future_idx})
    df_fc = m.predict(fut)[["ds", "yhat"]].rename(columns={"ds": "Week_Start"})
elif model_choice == "ARIMA":
    # Ensure weekly series aligned and fit ARIMA
    tmp = df_hist.set_index("Week_Start").asfreq("W-MON", fill_value=0)
    ar = ARIMA(tmp["y"], order=(1, 1, 1)).fit()
    pr = ar.get_forecast(steps=periods)
    df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": pr.predicted_mean.values})
else:
    # XGB placeholder: naive last observed value
    last_val = float(df_hist["y"].iloc[-1])
    df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})

# Round and cast
df_fc[forecast_label] = df_fc["yhat"].clip(lower=0).round(0).astype(int)

# ----------------------------
# Override with Amazon sell-out forecast where available
# ----------------------------
if up_path:
    df_up = read_amazon_forecast_csv(up_path)
    if df_up.empty:
        st.warning("⚠️ Amazon sell-out forecast file is empty or unreadable; skipping upstream merge.")
    else:
        df_fc["Week_Start"] = pd.to_datetime(df_fc["Week_Start"]).dt.to_period("W-MON").start_time
        df_fc = df_fc.merge(df_up, on="Week_Start", how="left")
        df_fc[forecast_label] = df_fc["Amazon_Sellout_Forecast"].fillna(df_fc[forecast_label]).astype(int)

# ----------------------------
# Projected sales ($)
# ----------------------------
df_fc["Projected_Sales"] = (df_fc[forecast_label] * float(unit_price)).round(2)

# ----------------------------
# Replenishment logic
# - Use recent average weekly demand for WOC target (more stable than per-week D)
# ----------------------------
hist_window = max(8, min(12, len(df_hist)))  # between 8 and 12 weeks depending on data
avg_weekly_demand = df_hist["y"].tail(hist_window).mean() if hist_window > 0 else df_hist["y"].mean()
avg_weekly_demand = float(avg_weekly_demand) if pd.notna(avg_weekly_demand) else 0.0

target_inventory = avg_weekly_demand * float(woc_target)

on_hand_begin = []
replenishments = []
prev = int(init_inv)

for _, r in df_fc.iterrows():
    demand = int(r[forecast_label])
    # Order enough to reach the target inventory *before* consuming this week's demand
    order_qty = max(int(round(target_inventory - prev)), 0)
    on_hand_begin.append(int(prev))
    replenishments.append(order_qty)
    # Inventory evolves: add order, then subtract demand
    prev = prev + order_qty - demand
    if prev < 0:
        prev = 0  # no backorders modeled here

df_fc["On_Hand_Begin"] = on_hand_begin
df_fc["Replenishment"] = replenishments
# Weeks of cover: avoid div-by-zero
df_fc["Weeks_Of_Cover"] = np.where(
    df_fc[forecast_label] > 0,
    (df_fc["On_Hand_Begin"] / df_fc[forecast_label]).round(2),
    np.nan
)

# Format date for display
df_fc["Date"] = pd.to_datetime(df_fc["Week_Start"]).dt.strftime("%d-%m-%Y")

# ----------------------------
# Plot: dual-axis (primary depends on Projection Type)
# ----------------------------
st.subheader(f"{periods}-Week Forecast & Replenishment")

# Decide which metric is primary
if projection_type == "Sales $":
    primary_series_key = "Projected_Sales"
    primary_title = "Sales $"
    secondary_series_key = forecast_label
    secondary_title = "Units"
else:
    primary_series_key = forecast_label
    primary_title = "Units"
    secondary_series_key = "Projected_Sales"
    secondary_title = "Sales $"

if PLOTLY_INSTALLED:
    fig = go.Figure()
    # Primary axis traces
    fig.add_trace(
        go.Scatter(
            x=df_fc["Week_Start"],
            y=df_fc[primary_series_key],
            name=f"{primary_title} ({'Projected' if primary_series_key=='Projected_Sales' else 'Forecast'})",
            yaxis="y",
            line=dict(color=AMZ_ORANGE),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_fc["Week_Start"],
            y=df_fc["Replenishment"],
            name="Replenishment Units",
            yaxis="y" if primary_series_key == forecast_label else "y2",
            line=dict(color=AMZ_BLUE),
        )
    )
    # Secondary axis trace (only if different from primary)
    if secondary_series_key != primary_series_key:
        fig.add_trace(
            go.Scatter(
                x=df_fc["Week_Start"],
                y=df_fc[secondary_series_key],
                name=f"{secondary_title} ({'Projected' if secondary_series_key=='Projected_Sales' else 'Forecast'})",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )

    fig.update_layout(
        xaxis=dict(title="Week"),
        yaxis=dict(title=primary_title),
        yaxis2=dict(title=secondary_title, overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(tickformat="%d-%m-%Y")
    st.plotly_chart(fig, use_container_width=True)
else:
    # Fallback: two simple Streamlit charts
    st.line_chart(df_fc.set_index("Week_Start")[[forecast_label, "Replenishment"]])
    st.line_chart(df_fc.set_index("Week_Start")["Projected_Sales"])

# ----------------------------
# Recap summary table
# ----------------------------
st.subheader("Summary Metrics")
total_rep = int(df_fc["Replenishment"].sum())
total_sales = float(df_fc["Projected_Sales"].sum())
avg_sales = float(df_fc["Projected_Sales"].mean())

recap = pd.DataFrame(
    {
        "Metric": ["Total Replenishment Units", "Total Projected Sales $", "Avg Weekly Sales $"],
        "Value": [f"{total_rep:,}", f"${total_sales:,.2f}", f"${avg_sales:,.2f}"],
    }
)
st.table(recap)

# ----------------------------
# Detailed table
# ----------------------------
st.subheader("Detailed Plan")
display_cols = ["Date", forecast_label, "Projected_Sales", "On_Hand_Begin", "Replenishment", "Weeks_Of_Cover"]
st.dataframe(df_fc[display_cols], use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    f"<div style='text-align:center;color:gray;margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True,
)
