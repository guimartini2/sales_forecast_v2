"""
Amazon Replenishment Forecast Streamlit App (Amazon Branded)

Hardened build:
- Robust loaders for Sales & Amazon Forecast (multi sep/skiprows, auto date/units column detection)
- Works with "Start - End" or single-date columns; picks the column with most valid dates
- Units column chosen by (name regex OR numeric dominance)
- Weekly alignment via W-MON with .dt.start_time (fixed)
- Continues even if history sums to 0 (shows warning, produces zero forecast)
- Proper Plotly dual-axis; Projection Type toggle drives primary axis
"""

import os
import re
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Optional libs
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
        from fbprophet import Prophet
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

# Branding
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# Page
st.set_page_config(page_title="Amazon Replenishment Forecast", page_icon="🛒", layout="wide")
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>", unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
fcst_file = st.sidebar.file_uploader("Amazon Sell-Out Forecast CSV", type=["csv"])
projection_type = st.sidebar.selectbox("Projection Type (primary Y-axis)", ["Units", "Sales $"])
init_inv = st.sidebar.number_input("Current On-Hand Inventory (units)", min_value=0, value=26730, step=1)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=10.0, step=0.01)

model_opts = []
if PROPHET_INSTALLED: model_opts.append("Prophet")
if ARIMA_INSTALLED:   model_opts.append("ARIMA")
if XGB_INSTALLED:     model_opts.append("XGBoost (naive)")
if not model_opts:
    st.error("Install at least one forecasting engine: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Forecast Model", model_opts)

woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = int(st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12))

st.markdown("---")
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to generate the forecast and replenishment plan.")
    st.stop()

# -------- Helpers --------
def _maybe_seek_start(obj):
    if hasattr(obj, "seek"):
        try: obj.seek(0)
        except Exception: pass

def _parse_date_series(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # If "Start - End" ranges, use the first part
    first = np.where(s.str.contains(" - "), s.str.split(" - ").str[0].str.strip(), s)
    return pd.to_datetime(first, errors="coerce")

def detect_date_column(df: pd.DataFrame):
    # Score columns by how many valid dates they produce
    scores = []
    for c in df.columns:
        dt = _parse_date_series(df[c])
        valid = dt.notna().sum()
        scores.append((c, valid, dt))
    best = max(scores, key=lambda x: x[1]) if scores else (None, 0, None)
    if best[1] < 4:
        return None, None  # not enough dates to be credible
    return best[0], best[2]

def detect_units_column(df: pd.DataFrame, date_col: str):
    # Prefer regex-named columns
    regex_candidates = [c for c in df.columns if c != date_col and re.search(r"(unit|qty|quantity)", str(c), re.I)]
    numeric_strength = {}
    for c in df.columns:
        if c == date_col: continue
        vals = pd.to_numeric(df[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        numeric_strength[c] = vals.abs().sum(skipna=True)
    # Pick best among regex by numeric sum; else best overall numeric
    if regex_candidates:
        best_regex = max(regex_candidates, key=lambda c: numeric_strength.get(c, -1))
        if numeric_strength.get(best_regex, 0) > 0:
            return best_regex
    # Fallback: any numeric-ish column with the highest sum
    if numeric_strength:
        best_any = max(numeric_strength, key=numeric_strength.get)
        return best_any
    return None

def read_sales_csv(src):
    # Try multiple separators/skiprows combos
    for skip in (0, 1, 2, 3):
        for sep in (",", ";", "\t", "|"):
            try:
                _maybe_seek_start(src)
                t = pd.read_csv(src, skiprows=skip, sep=sep, engine="python")
                if t.empty or t.shape[1] < 1:
                    continue
                date_col, parsed = detect_date_column(t)
                if date_col is None:
                    continue
                t = t.copy()
                t["Week_Start"] = parsed
                unit_col = detect_units_column(t, date_col="Week_Start")
                if unit_col is None:
                    continue
                # Build clean frame
                vals = pd.to_numeric(t[unit_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce").fillna(0)
                out = pd.DataFrame({"Week_Start": t["Week_Start"], "y": vals})
                out = out.dropna(subset=["Week_Start"])
                # Normalize to Monday week
                out["Week_Start"] = pd.to_datetime(out["Week_Start"]).dt.to_period("W-MON").dt.start_time
                out = out.groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
                # Pass back metadata for debug
                return out, date_col, unit_col, sep, skip
            except Exception:
                continue
    return None, None, None, None, None

def try_parse_any_date(text):
    s = str(text).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d): return d
    return pd.to_datetime(s, errors="coerce")

def read_amazon_forecast_csv(src):
    df_raw = None
    for skip in (0, 1, 2):
        for sep in (",", ";", "\t", "|"):
            try:
                _maybe_seek_start(src)
                tmp = pd.read_csv(src, sep=sep, skiprows=skip, engine="python")
                if not tmp.empty and tmp.shape[1] > 1:
                    df_raw = tmp; break
            except Exception:
                continue
        if df_raw is not None: break

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"])

    row0 = df_raw.iloc[0]
    rec, year_now = [], datetime.now().year
    for col in df_raw.columns:
        s = str(col)
        if "(" in s and ")" in s:
            inside = s.split("(")[-1].split(")")[0].strip()
            dt = try_parse_any_date(f"{inside} {year_now}") or try_parse_any_date(inside)
        else:
            dt = try_parse_any_date(s)
        if pd.notna(dt):
            val = pd.to_numeric(str(row0[col]).replace(",", ""), errors="coerce")
            if pd.notna(val):
                rec.append({"Week_Start": dt, "Amazon_Sellout_Forecast": int(round(val))})
    if not rec:
        return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"])

    df_up = pd.DataFrame(rec).dropna(subset=["Week_Start"])
    df_up["Week_Start"] = pd.to_datetime(df_up["Week_Start"]).dt.to_period("W-MON").dt.start_time
    df_up = df_up.groupby("Week_Start", as_index=False)["Amazon_Sellout_Forecast"].sum().sort_values("Week_Start")
    return df_up

def future_weeks(start_date, periods):
    start_date = pd.to_datetime(start_date)
    # Next Monday strictly after start_date
    next_mon = (start_date + pd.offsets.Week(weekday=0))
    if next_mon <= start_date:
        next_mon = next_mon + pd.offsets.Week(weekday=0)
    return pd.date_range(start=next_mon, periods=periods, freq="W-MON")

# -------- Load files --------
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_up    = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

sales_path = sales_file if sales_file is not None else (default_sales if os.path.exists(default_sales) else None)
up_path    = fcst_file  if fcst_file  is not None else (default_up    if os.path.exists(default_up)    else None)

if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

df_hist, detected_date_col, detected_units_col, used_sep, used_skip = read_sales_csv(sales_path)
if df_hist is None or df_hist.empty:
    st.error("Could not parse Sales history CSV (no valid dates/units after normalization).")
    st.stop()

today = pd.to_datetime(datetime.now().date())
df_hist = df_hist[df_hist["Week_Start"] <= today]

# Debug info (optional, won’t break anything)
with st.expander("Debug: Detected columns / parse choices"):
    meta = {
        "Detected date column": detected_date_col,
        "Detected units column": detected_units_col,
        "CSV separator used": used_sep,
        "Skiprows used": used_skip,
        "History rows (post-normalization)": int(len(df_hist)),
        "History sum(y)": float(df_hist["y"].sum()),
    }
    st.json(meta)

# -------- Forecast horizon --------
if df_hist.empty:
    st.warning("Historical data is empty after filtering to today. Proceeding with zero baseline.")
last_hist_week = df_hist["Week_Start"].max() if not df_hist.empty else today
future_idx = future_weeks(max(last_hist_week, today), periods)

# -------- Forecast generation --------
forecast_label = "Forecast_Units"
if model_choice == "Prophet" and PROPHET_INSTALLED and not df_hist.empty and df_hist["y"].sum() > 0:
    m = Prophet(weekly_seasonality=True)
    m.fit(df_hist.rename(columns={"Week_Start": "ds", "y": "y"}))
    fut = pd.DataFrame({"ds": future_idx})
    df_fc = m.predict(fut)[["ds", "yhat"]].rename(columns={"ds": "Week_Start"})
elif model_choice == "ARIMA" and ARIMA_INSTALLED:
    tmp = df_hist.set_index("Week_Start").asfreq("W-MON", fill_value=0)
    series = tmp["y"] if not tmp.empty else pd.Series([0.0], index=pd.date_range(today, periods=1, freq="W-MON"))
    try:
        ar = ARIMA(series, order=(1, 1, 1)).fit()
        pr = ar.get_forecast(steps=periods)
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": pr.predicted_mean.values})
    except Exception:
        # Fallback to naive if ARIMA fails
        last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})
else:
    last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
    df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})

df_fc[forecast_label] = df_fc["yhat"].clip(lower=0).round(0).astype(int)

# -------- Override with Amazon sell-out forecast --------
if up_path:
    df_up = read_amazon_forecast_csv(up_path)
    if df_up.empty:
        st.warning("⚠️ Amazon sell-out forecast file is empty or unreadable; skipping upstream merge.")
    else:
        df_fc["Week_Start"] = pd.to_datetime(df_fc["Week_Start"]).dt.to_period("W-MON").dt.start_time
        df_fc = df_fc.merge(df_up, on="Week_Start", how="left")
        df_fc[forecast_label] = df_fc["Amazon_Sellout_Forecast"].fillna(df_fc[forecast_label]).astype(int)

# -------- Sales $ --------
df_fc["Projected_Sales"] = (df_fc[forecast_label] * float(unit_price)).round(2)

# -------- Replenishment (WOC on recent average) --------
hist_window = max(8, min(12, len(df_hist))) if not df_hist.empty else 0
avg_weekly_demand = (df_hist["y"].tail(hist_window).mean() if hist_window > 0 else 0.0)
avg_weekly_demand = float(avg_weekly_demand) if pd.notna(avg_weekly_demand) else 0.0
target_inventory = avg_weekly_demand * float(woc_target)

on_hand_begin, replenishments = [], []
prev = int(init_inv)
for _, r in df_fc.iterrows():
    demand = int(r[forecast_label])
    order_qty = max(int(round(target_inventory - prev)), 0)
    on_hand_begin.append(int(prev))
    replenishments.append(order_qty)
    prev = max(prev + order_qty - demand, 0)

df_fc["On_Hand_Begin"] = on_hand_begin
df_fc["Replenishment"] = replenishments
df_fc["Weeks_Of_Cover"] = np.where(
    df_fc[forecast_label] > 0,
    (df_fc["On_Hand_Begin"] / df_fc[forecast_label]).round(2),
    np.nan
)
df_fc["Date"] = pd.to_datetime(df_fc["Week_Start"]).dt.strftime("%d-%m-%Y")

# Warn if history is zero
if df_hist["y"].sum() == 0:
    st.warning("Historical units sum to 0 after parsing. Forecast defaults to zero; adjust data or unit column if this seems wrong.")

# -------- Plot --------
st.subheader(f"{periods}-Week Forecast & Replenishment")

if projection_type == "Sales $":
    primary_key, primary_title = "Projected_Sales", "Sales $"
    secondary_key, secondary_title = forecast_label, "Units"
else:
    primary_key, primary_title = forecast_label, "Units"
    secondary_key, secondary_title = "Projected_Sales", "Sales $"

if PLOTLY_INSTALLED:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc[primary_key],
                             name=f"{primary_title} ({'Projected' if primary_key=='Projected_Sales' else 'Forecast'})",
                             yaxis="y", line=dict(color=AMZ_ORANGE)))
    # Put replenishment on units axis if primary is units, else on secondary
    fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc["Replenishment"],
                             name="Replenishment Units",
                             yaxis="y" if primary_key == forecast_label else "y2",
                             line=dict(color=AMZ_BLUE)))
    if secondary_key != primary_key:
        fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc[secondary_key],
                                 name=f"{secondary_title} ({'Projected' if secondary_key=='Projected_Sales' else 'Forecast'})",
                                 yaxis="y2", line=dict(dash="dot")))
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
    st.line_chart(df_fc.set_index("Week_Start")[[forecast_label, "Replenishment"]])
    st.line_chart(df_fc.set_index("Week_Start")["Projected_Sales"])

# -------- Summary --------
st.subheader("Summary Metrics")
total_rep = int(df_fc["Replenishment"].sum())
total_sales = float(df_fc["Projected_Sales"].sum())
avg_sales = float(df_fc["Projected_Sales"].mean())
recap = pd.DataFrame({
    "Metric": ["Total Replenishment Units", "Total Projected Sales $", "Avg Weekly Sales $"],
    "Value": [f"{total_rep:,}", f"${total_sales:,.2f}", f"${avg_sales:,.2f}"]
})
st.table(recap)

# -------- Detail --------
st.subheader("Detailed Plan")
st.dataframe(df_fc[["Date", forecast_label, "Projected_Sales", "On_Hand_Begin", "Replenishment", "Weeks_Of_Cover"]],
             use_container_width=True)

st.markdown(f"<div style='text-align:center;color:gray;margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>", unsafe_allow_html=True)
