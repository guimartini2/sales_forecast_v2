"""
Amazon Replenishment Forecast — Wide/Long CSV Compatible
(Only targeted fixes)
- Sales loader: detect "long" table with 'Week' column and parse Shipped/Ordered Units
- Amazon loader: search skiprows to find week columns (e.g., "Week 0 (3 Aug - 9 Aug)")
- Both loaders produce Week_Start and numeric values; alignment to W-MON
- Override uses outer merge on union of weeks
"""

import os
import re
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

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

# Branding/colors
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# Page
st.set_page_config(page_title="Amazon Replenishment Forecast", page_icon="🛒", layout="wide")
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV (Amazon export)", type=["csv"])
fcst_file = st.sidebar.file_uploader("Amazon Sell-Out Forecast CSV (wide)", type=["csv"])

projection_type = st.sidebar.selectbox("Projection Type (primary Y-axis)", ["Units", "Sales $"])
init_inv = st.sidebar.number_input("Current On-Hand Inventory (units)", min_value=0, value=26730, step=1)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=10.0, step=0.01)

model_opts = []
if PROPHET_INSTALLED: model_opts.append("Prophet")
if ARIMA_INSTALLED:   model_opts.append("ARIMA")
if not model_opts:    model_opts.append("Naive (last value)")
model_choice = st.sidebar.selectbox("Forecast Model", model_opts)

woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = int(st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12))

st.markdown("---")
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to generate the forecast and replenishment plan.")
    st.stop()

# Defaults (paths from your session)
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2025_8-6-2025.csv"
default_up    = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
sales_path = sales_file if sales_file is not None else (default_sales if os.path.exists(default_sales) else None)
up_path    = fcst_file  if fcst_file  is not None else (default_up    if os.path.exists(default_up)    else None)

if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

# ------------------------ Helpers (unchanged) ------------------------
def _maybe_seek_start(obj) -> None:
    if hasattr(obj, "seek"):
        try: obj.seek(0)
        except Exception: pass

def try_parse_date_string(s: str, prefer_year: Optional[int] = None) -> Optional[pd.Timestamp]:
    s = str(s).strip()
    if prefer_year and re.search(r"\b\d{4}\b", s) is None and re.search(r"\b\d{2}\b", s) is None:
        s_forced = f"{s} {prefer_year}"
    else:
        s_forced = s
    for fmt in ("%d %b %Y", "%d %B %Y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%d-%m-%Y"):
        dt = pd.to_datetime(s_forced, format=fmt, errors="coerce")
        if pd.notna(dt):
            return pd.to_datetime(dt)
    dt = pd.to_datetime(s_forced, errors="coerce")
    if pd.notna(dt):
        return pd.to_datetime(dt)
    return None

def extract_weekstart_from_header(col: str, prefer_year: Optional[int] = None) -> Optional[pd.Timestamp]:
    s = str(col)
    m = re.search(r"\(([^)]*)\)", s)
    candidate = m.group(1) if m else s
    wk = re.search(r"Week of\s*(.*)$", s, flags=re.I)
    if wk:
        candidate = wk.group(1).strip()
    parts = re.split(r"\s*-\s*", candidate.strip())
    first_part = parts[0] if parts else candidate.strip()
    first_part = re.sub(r"^Week\s*\d+\s*", "", first_part, flags=re.I).strip()
    dt = try_parse_date_string(first_part, prefer_year=prefer_year)
    if dt is None:
        m2 = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", candidate)
        if m2:
            dt = try_parse_date_string(m2.group(1), prefer_year=prefer_year)
        else:
            m3 = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9})\b", candidate)
            if m3:
                dt = try_parse_date_string(m3.group(1), prefer_year=prefer_year)
    if dt is None:
        return None
    return pd.to_datetime(dt).to_period("W-MON").start_time

def future_weeks_after(start_date, periods: int) -> pd.DatetimeIndex:
    start_date = pd.to_datetime(start_date)
    next_mon = (start_date + pd.offsets.Week(weekday=0))
    if next_mon <= start_date:
        next_mon = next_mon + pd.offsets.Week(weekday=0)
    return pd.date_range(start=next_mon, periods=periods, freq="W-MON")

# ------------------------ SALES LOADER (modified) ------------------------
def read_sales_to_long(src) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle Amazon Retail 'View By = Week' export:
      Row 0: metadata line
      Row 1: header with 'Week', 'Ordered Units', 'Shipped Units', etc.
      Rows: "YYYY-MM-DD - YYYY-MM-DD", values like "3,751"
    """
    # Try common skiprows to expose the real header (we know 1 works on your file)
    for skip in (1, 0, 2, 3, 4):
        try:
            _maybe_seek_start(src)
            df = pd.read_csv(src, sep=None, engine="python", skiprows=skip)
            if "Week" not in df.columns:
                continue
            # Prefer Shipped Units, else Ordered Units, else any *Units*
            units_col = None
            for cand in ["Shipped Units", "Ordered Units"]:
                if cand in df.columns:
                    units_col = cand; break
            if units_col is None:
                for c in df.columns:
                    if re.search(r"Units", str(c), re.I):
                        units_col = c; break
            if units_col is None:
                continue

            # Parse start of week from "YYYY-MM-DD - YYYY-MM-DD"
            week_start = pd.to_datetime(df["Week"].astype(str).str.split(" - ").str[0], errors="coerce")
            vals = pd.to_numeric(df[units_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce").fillna(0)
            out = pd.DataFrame({"Week_Start": week_start.dt.to_period("W-MON").dt.start_time, "y": vals})
            out = out.dropna(subset=["Week_Start"]).groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
            meta = {"mode": "long", "skip": skip, "units_col": units_col, "rows": int(len(out)), "sum_y": float(out["y"].sum())}
            return out, meta
        except Exception:
            continue

    # Fallback (rare): treat as wide with week columns (kept from previous version)
    _maybe_seek_start(src)
    df = pd.read_csv(src, sep=None, engine="python")
    prefer_year = datetime.now().year
    week_cols, week_map = [], {}
    for c in df.columns:
        wk = extract_weekstart_from_header(c, prefer_year)
        if wk is not None:
            week_cols.append(c)
            week_map[c] = wk
    if not week_cols:
        st.error("Sales CSV: No usable 'Week' column or week headers found.")
        st.stop()
    id_cols = [c for c in df.columns if c not in week_cols]
    long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="col", value_name="val")
    long["Week_Start"] = long["col"].map(week_map)
    long.dropna(subset=["Week_Start"], inplace=True)
    long["y"] = pd.to_numeric(long["val"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce").fillna(0)
    out = long.groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
    meta = {"mode": "wide", "rows": int(len(out)), "sum_y": float(out["y"].sum())}
    return out, meta

# ------------------------ AMAZON FORECAST LOADER (modified) ------------------------
def read_amazon_forecast_to_long(src) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Find the header row with 'Week N (dd Mon - dd Mon)' columns by scanning skiprows.
    Melt to long and sum across rows (if multiple ASINs).
    """
    prefer_year = datetime.now().year
    for skip in range(0, 6):
        try:
            _maybe_seek_start(src)
            df = pd.read_csv(src, sep=None, engine="python", skiprows=skip)
            week_cols = [c for c in df.columns if re.search(r"Week\s*\d+\s*\(", str(c))]
            if not week_cols:
                continue
            id_cols = [c for c in df.columns if c not in week_cols]
            long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="col", value_name="val")
            long["Week_Start"] = long["col"].apply(lambda c: extract_weekstart_from_header(c, prefer_year))
            long.dropna(subset=["Week_Start"], inplace=True)
            long["nval"] = pd.to_numeric(long["val"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
            df_up = long.groupby("Week_Start", as_index=False)["nval"].sum().rename(columns={"nval": "Amazon_Sellout_Forecast"})
            df_up["Amazon_Sellout_Forecast"] = df_up["Amazon_Sellout_Forecast"].round().astype(int)
            df_up = df_up.sort_values("Week_Start")
            meta = {"skip": skip, "week_cols_found": len(week_cols), "rows": int(len(df_up))}
            return df_up, meta
        except Exception:
            continue
    return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"]), {"skip": None, "week_cols_found": 0, "rows": 0}

# ------------------------ Load & reshape ------------------------
df_hist, sales_meta = read_sales_to_long(sales_path)
today = pd.to_datetime(datetime.now().date())
df_hist_filtered = df_hist[df_hist["Week_Start"] <= today]
df_hist = df_hist_filtered if not df_hist_filtered.empty else df_hist

df_up = pd.DataFrame()
up_meta = {"week_cols_found": 0}
if up_path:
    df_up, up_meta = read_amazon_forecast_to_long(up_path)

with st.expander("Debug: parsing summary"):
    st.json({"sales_meta": sales_meta, "amazon_forecast_meta": up_meta,
             "hist_rows": int(len(df_hist)), "hist_sum": float(df_hist["y"].sum())})

# ------------------------ Forecast (unchanged) ------------------------
forecast_label = "Forecast_Units"
last_week = df_hist["Week_Start"].max() if not df_hist.empty else today
future_idx = future_weeks_after(max(last_week, today), periods)

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
        last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})
else:
    last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
    df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})

df_fc[forecast_label] = pd.Series(df_fc["yhat"]).clip(lower=0).round().astype(int)

# ------------------------ Override with Amazon forecast (outer merge; unchanged logic) ------------------------
if not df_up.empty:
    merged = pd.merge(df_fc[["Week_Start", forecast_label]], df_up, on="Week_Start", how="outer")
    merged = merged.sort_values("Week_Start").reset_index(drop=True)
    merged[forecast_label] = merged["Amazon_Sellout_Forecast"].fillna(merged[forecast_label]).fillna(0).astype(int)
    start_from = future_weeks_after(max(last_week, today), 1)[0]
    horizon = pd.date_range(start=start_from, periods=periods, freq="W-MON")
    # Align all to Monday, then reindex to horizon
    merged["Week_Start"] = pd.to_datetime(merged["Week_Start"]).to_period("W-MON").dt.start_time
    merged = merged.groupby("Week_Start", as_index=False)[forecast_label].sum()
    merged = merged.set_index("Week_Start").reindex(horizon, fill_value=0).reset_index().rename(columns={"index": "Week_Start"})
    df_fc = merged.copy()
else:
    st.warning("⚠️ Amazon sell-out forecast file parsed with 0 week columns or was missing; not overriding.")

# ------------------------ Projected $ + Replenishment (unchanged) ------------------------
df_fc["Projected_Sales"] = (df_fc[forecast_label] * float(unit_price)).round(2)

hist_window = max(8, min(12, len(df_hist))) if not df_hist.empty else 0
avg_weekly_demand = (df_hist["y"].tail(hist_window).mean() if hist_window > 0 else 0.0)
avg_weekly_demand = float(avg_weekly_demand) if pd.notna(avg_weekly_demand) else 0.0
target_inventory = avg_weekly_demand * float(woc_target)

on_hand_begin, replenishments = [], []
prev = int(init_inv)
for _, r in df_fc.ite
