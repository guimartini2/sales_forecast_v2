"""
Amazon Replenishment Forecast — Predict Weekly Amazon POs (Sell-In)

Modified to read .xlsx files only instead of .csv files.

Fix in this revision:
- Make Amazon Sell-Out Excel the **authoritative** source of Forecast_Units.
  When df_up (Amazon) is non-empty, we IGNORE model output and build the horizon
  strictly from Amazon's future weeks (trimmed to `periods` if needed).
- Prior fixes kept: sales Excel parsing, 2nd-row Amazon header fallback,
  no rendering of model objects in Streamlit.
"""

import os
import re
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------ Optional libs ------------------------
PLOTLY_INSTALLED = False
try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    pass

PROPHET_INSTALLED = False
try:
    from prophet import Prophet  # noqa: F401
    PROPHET_INSTALLED = True
except ImportError:
    try:
        from fbprophet import Prophet  # noqa: F401
        PROPHET_INSTALLED = True
    except ImportError:
        pass

ARIMA_INSTALLED = False
try:
    from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
    ARIMA_INSTALLED = True
except ImportError:
    pass

# ------------------------ Branding ------------------------
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# ------------------------ Page ------------------------
st.set_page_config(page_title="Amazon Replenishment Forecast", page_icon="🛒", layout="wide")
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>",
    unsafe_allow_html=True,
)

# ------------------------ Sidebar ------------------------
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history Excel (Amazon export)", type=["xlsx"])
fcst_file  = st.sidebar.file_uploader("Amazon Sell-Out Forecast Excel", type=["xlsx"])
inv_file   = st.sidebar.file_uploader("Amazon Inventory Excel (optional, weekly On Hand)", type=["xlsx"])

projection_type = st.sidebar.selectbox("Projection Type (primary Y-axis)", ["Units", "Sales $"])
init_inv = st.sidebar.number_input("Fallback Current On-Hand Inventory (units)", min_value=0, value=26730, step=1)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=10.0, step=0.01)

model_opts = []
if PROPHET_INSTALLED: model_opts.append("Prophet")
if ARIMA_INSTALLED:   model_opts.append("ARIMA")
if not model_opts:    model_opts.append("Naive (last value)")
model_choice = st.sidebar.selectbox("Forecast Model (used ONLY if Amazon file missing)", model_opts)

woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = int(st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12))
lead_time_weeks = int(st.sidebar.number_input("Lead Time (weeks) — PO arrival delay", min_value=0, max_value=26, value=2))

st.markdown("---")
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to generate the forecast and predicted POs.")
    st.stop()

# ------------------------ Defaults ------------------------
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.xlsx"
default_up    = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.xlsx"
sales_path = sales_file if sales_file is not None else (default_sales if os.path.exists(default_sales) else None)
up_path    = fcst_file  if fcst_file  is not None else (default_up    if os.path.exists(default_up)    else None)
inv_path   = inv_file   if inv_file   is not None else None

if not sales_path:
    st.error("Sales history Excel file is required.")
    st.stop()

# ------------------------ Helpers ------------------------
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

# ------------------------ SALES LOADER (Excel format) ------------------------
def read_sales_to_long(src) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        _maybe_seek_start(src)
        # Try reading Excel file with different sheet names and skip rows
        for sheet_name in [None, 0, "Sales", "Data", "Sheet1"]:
            for skip_rows in [0, 1, 2]:
                try:
                    df = pd.read_excel(src, sheet_name=sheet_name, skiprows=skip_rows)
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    # Look for week column and units column
                    week_col = None
                    units_col = None
                    
                    # Find week column (first column or one containing "week")
                    if len(df.columns) > 0:
                        week_col = df.columns[0]
                        for col in df.columns:
                            if re.search(r"week", str(col), re.I):
                                week_col = col
                                break
                    
                    # Find units column ("Ordered Units" or 3rd column)
                    if "Ordered Units" in df.columns:
                        units_col = "Ordered Units"
                    elif len(df.columns) >= 3:
                        units_col = df.columns[2]
                    else:
                        continue
                    
                    if week_col and units_col and len(df) > 0:
                        # Parse week starts - handle date ranges like "Week 1 - 01/01/2024"
                        week_data = df[week_col].astype(str)
                        week_start = []
                        
                        for week_str in week_data:
                            if pd.isna(week_str) or str(week_str).strip() == 'nan':
                                week_start.append(pd.NaT)
                                continue
                            
                            # Try to extract date from various formats
                            if " - " in str(week_str):
                                date_part = str(week_str).split(" - ")[-1].strip()
                            else:
                                date_part = str(week_str).strip()
                            
                            parsed_date = pd.to_datetime(date_part, errors="coerce")
                            week_start.append(parsed_date)
                        
                        week_start = pd.Series(week_start)
                        
                        # Parse units - remove non-numeric characters
                        units = pd.to_numeric(
                            df[units_col].astype(str).str.replace(r"[^0-9\-.]", "", regex=True), 
                            errors="coerce"
                        ).fillna(0)
                        
                        out = pd.DataFrame({
                            "Week_Start": week_start.dt.to_period("W-MON").dt.start_time, 
                            "y": units
                        })
                        out = out.dropna(subset=["Week_Start"]).groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
                        
                        if len(out) > 0:  # Successfully parsed data
                            meta = {
                                "mode": "long", 
                                "format": "excel", 
                                "rows": int(len(out)), 
                                "sum_y": float(out["y"].sum()),
                                "week_col": week_col, 
                                "units_col": units_col,
                                "sheet": sheet_name,
                                "skip_rows": skip_rows
                            }
                            return out, meta
                except Exception:
                    continue
    except Exception:
        pass

    # Fallback: wide header melt (for Excel files with week columns as headers)
    try:
        _maybe_seek_start(src)
        for sheet_name in [None, 0, "Sales", "Data", "Sheet1"]:
            try:
                df = pd.read_excel(src, sheet_name=sheet_name)
                prefer_year = datetime.now().year
                week_cols, week_map = [], {}
                
                for c in df.columns:
                    wk = extract_weekstart_from_header(str(c), prefer_year)
                    if wk is not None:
                        week_cols.append(c)
                        week_map[c] = wk
                
                if week_cols:
                    id_cols = [c for c in df.columns if c not in week_cols]
                    long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="col", value_name="val")
                    long["Week_Start"] = long["col"].map(week_map)
                    long.dropna(subset=["Week_Start"], inplace=True)
                    long["y"] = pd.to_numeric(
                        long["val"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), 
                        errors="coerce"
                    ).fillna(0)
                    out = long.groupby("Week_Start", as_index=False)["y"].sum().sort_values("Week_Start")
                    meta = {"mode": "wide", "rows": int(len(out)), "sum_y": float(out["y"].sum()), "sheet": sheet_name}
                    return out, meta
            except Exception:
                continue
    except Exception:
        pass
    
    st.error("Sales Excel: No usable 'Week' column or week headers found.")
    st.stop()

# ------------------------ AMAZON FORECAST LOADER (Excel format) ------------------------
def read_amazon_forecast_to_long(src) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    prefer_year = datetime.now().year
    
    # Try different sheets and skip rows
    for sheet_name in [None, 0, "Forecast", "Data", "Sheet1"]:
        for skip in range(0, 6):
            try:
                _maybe_seek_start(src)
                df = pd.read_excel(src, sheet_name=sheet_name, skiprows=skip)
                
                week_cols = [c for c in df.columns if re.search(r"Week\s*\d+\s*\(", str(c))]
                if not week_cols:
                    continue
                
                id_cols = [c for c in df.columns if c not in week_cols]
                long = df.melt(id_vars=id_cols, value_vars=week_cols, var_name="col", value_name="val")
                long["Week_Start"] = long["col"].apply(lambda c: extract_weekstart_from_header(str(c), prefer_year))
                long.dropna(subset=["Week_Start"], inplace=True)
                long["nval"] = pd.to_numeric(
                    long["val"].astype(str).str.replace(r"[^0-9\-.]", "", regex=True), 
                    errors="coerce"
                )
                df_up = long.groupby("Week_Start", as_index=False)["nval"].sum().rename(columns={"nval": "Amazon_Sellout_Forecast"})
                df_up["Amazon_Sellout_Forecast"] = df_up["Amazon_Sellout_Forecast"].round().astype(int)
                df_up = df_up.sort_values("Week_Start")
                
                meta = {
                    "skip": skip, 
                    "mode": "normal", 
                    "week_cols_found": len(week_cols), 
                    "rows": int(len(df_up)),
                    "sum": int(df_up["Amazon_Sellout_Forecast"].sum()) if not df_up.empty else 0,
                    "sheet": sheet_name
                }
                return df_up, meta
            except Exception:
                continue
    
    # Fallback: 2nd-row headers, units on 3rd row+
    for sheet_name in [None, 0, "Forecast", "Data", "Sheet1"]:
        try:
            _maybe_seek_start(src)
            raw = pd.read_excel(src, sheet_name=sheet_name, header=None)
            
            if raw.shape[0] >= 3:
                header = raw.iloc[1].astype(str).tolist()
                data = raw.iloc[2:].copy()
                data.columns = header
                
                week_cols = [c for c in data.columns if re.search(r"Week\s*\d+\s*\(|\(\d{1,2}\s*[A-Za-z]", str(c))]
                if week_cols:
                    vals = data[week_cols].applymap(lambda x: re.sub(r"[^0-9\-.]", "", str(x)))
                    vals = vals.apply(pd.to_numeric, errors="coerce")
                    sums = vals.sum(axis=0, skipna=True)
                    
                    recs = []
                    for col, val in sums.items():
                        dt = extract_weekstart_from_header(str(col), prefer_year)
                        if dt is not None and pd.notna(val):
                            recs.append({
                                "Week_Start": pd.to_datetime(dt).to_period("W-MON").start_time,
                                "Amazon_Sellout_Forecast": int(round(val))
                            })
                    
                    df_up = pd.DataFrame(recs).sort_values("Week_Start")
                    meta = {
                        "mode": "2nd_row_header", 
                        "week_cols_found": len(week_cols), 
                        "rows": int(len(df_up)),
                        "sum": int(df_up["Amazon_Sellout_Forecast"].sum()) if not df_up.empty else 0,
                        "sheet": sheet_name
                    }
                    return df_up, meta
        except Exception:
            continue
    
    return pd.DataFrame(columns=["Week_Start", "Amazon_Sellout_Forecast"]), {"mode": "none", "week_cols_found": 0, "rows": 0, "sum": 0}

# ------------------------ INVENTORY LOADER (Excel format) ------------------------
def read_inventory_onhand(src) -> Optional[int]:
    today = pd.to_datetime(datetime.now().date())
    
    for sheet_name in [None, 0, "Inventory", "Data", "Sheet1"]:
        for skip in (0, 1, 2, 3, 4, 5):
            try:
                _maybe_seek_start(src)
                df = pd.read_excel(src, sheet_name=sheet_name, skiprows=skip)
                
                week_col = None
                for c in df.columns:
                    if re.search(r"^\s*week\s*$", str(c), re.I):
                        week_col = c
                        break
                
                if week_col is None:
                    continue
                
                onhand_col = None
                for c in df.columns:
                    if re.search(r"on\s*hand|onhand|o/h|inventory", str(c), re.I):
                        onhand_col = c
                        break
                
                if onhand_col is None:
                    continue
                
                # Parse week data - handle date ranges
                week_data = df[week_col].astype(str)
                wk_parsed = []
                for week_str in week_data:
                    if pd.isna(week_str) or str(week_str).strip() == 'nan':
                        wk_parsed.append(pd.NaT)
                        continue
                    
                    if " - " in str(week_str):
                        date_part = str(week_str).split(" - ")[0].strip()
                    else:
                        date_part = str(week_str).strip()
                    
                    wk_parsed.append(pd.to_datetime(date_part, errors="coerce"))
                
                wk = pd.Series(wk_parsed)
                oh = pd.to_numeric(
                    df[onhand_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), 
                    errors="coerce"
                )
                
                tmp = pd.DataFrame({
                    "Week_Start": wk.dt.to_period("W-MON").dt.start_time, 
                    "OH": oh
                })
                tmp = tmp.dropna(subset=["Week_Start", "OH"])
                tmp = tmp[tmp["Week_Start"] <= today].sort_values("Week_Start")
                
                if not tmp.empty:
                    return int(tmp["OH"].iloc[-1])
            except Exception:
                continue
    
    return None

# ------------------------ Load & reshape ------------------------
df_hist, sales_meta = read_sales_to_long(sales_path)
today = pd.to_datetime(datetime.now().date())
df_hist_filtered = df_hist[df_hist["Week_Start"] <= today]
df_hist = df_hist_filtered if not df_hist_filtered.empty else df_hist

df_up = pd.DataFrame()
up_meta = {"week_cols_found": 0}
if up_path:
    df_up, up_meta = read_amazon_forecast_to_long(up_path)

init_inv_override = None
if inv_path:
    init_inv_override = read_inventory_onhand(inv_path)
start_on_hand = int(init_inv_override) if init_inv_override is not None else int(init_inv)

# (Keep small debug, can collapse)
with st.expander("Debug: parsing summary", expanded=False):
    st.json({
        "sales_meta": sales_meta,
        "amazon_forecast_meta": up_meta,
        "start_on_hand_used": start_on_hand,
        "hist_rows": int(len(df_hist)),
        "hist_sum": float(df_hist["y"].sum())
    })

# ------------------------ Forecast (MODEL = FALLBACK ONLY) ------------------------
forecast_label = "Forecast_Units"
last_week = df_hist["Week_Start"].max() if not df_hist.empty else today
future_idx = future_weeks_after(max(last_week, today), periods)

if model_choice == "Prophet" and PROPHET_INSTALLED and not df_hist.empty and df_hist["y"].sum() > 0:
    m = Prophet(weekly_seasonality=True)  # type: ignore[name-defined]
    m.fit(df_hist.rename(columns={"Week_Start": "ds", "y": "y"}))
    fut = pd.DataFrame({"ds": future_idx})
    df_fc = m.predict(fut)[["ds", "yhat"]].rename(columns={"ds": "Week_Start"})
elif model_choice == "ARIMA" and ARIMA_INSTALLED:
    tmp = df_hist.set_index("Week_Start").asfreq("W-MON", fill_value=0)
    series = tmp["y"] if not tmp.empty else pd.Series([0.0], index=pd.date_range(today, periods=1, freq="W-MON"))
    try:
        ar = ARIMA(series, order=(1, 1, 1)).fit()  # type: ignore[name-defined]
        pr = ar.get_forecast(steps=periods)
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": pr.predicted_mean.values})
    except Exception:
        last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
        df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})
else:
    last_val = float(df_hist["y"].iloc[-1]) if not df_hist.empty else 0.0
    df_fc = pd.DataFrame({"Week_Start": future_idx, "yhat": np.full(len(future_idx), last_val)})

df_fc[forecast_label] = pd.Series(df_fc.get("yhat", 0)).clip(lower=0).round().astype(int)
df_fc["Week_Start"] = pd.to_datetime(df_fc["Week_Start"]).dt.to_period("W-MON").dt.start_time

# ------------------------ AMAZON = AUTHORITATIVE FORECAST ------------------------
amazon_drives = False
if not df_up.empty:
    amazon = df_up.copy()
    amazon["Week_Start"] = pd.to_datetime(amazon["Week_Start"]).dt.to_period("W-MON")
    # Future weeks only, sorted
    start_from = future_weeks_after(max(last_week, today), 1)[0].to_period("W-MON")
    amazon = amazon[amazon["Week_Start"] >= start_from].sort_values("Week_Start")
    if not amazon.empty:
        amazon = amazon.head(periods)  # respect selected horizon
        df_fc = amazon.rename(columns={"Amazon_Sellout_Forecast": forecast_label})[["Week_Start", forecast_label]]
        df_fc["Week_Start"] = df_fc["Week_Start"].dt.start_time
        amazon_drives = True

# ------------------------ Projected $ ------------------------
df_fc["Projected_Sales"] = (df_fc[forecast_label] * float(unit_price)).round(2)

# ------------------------ Predict Weekly POs (Order-Up-To with lead time) ------------------------
# Demand baseline MUST be Forecast_Units (now = Amazon when available)
hist_window = max(8, min(12, len(df_hist))) if not df_hist.empty else 0
avg_weekly_demand = (df_hist["y"].tail(hist_window).mean() if hist_window > 0 else 0.0)
avg_weekly_demand = float(avg_weekly_demand) if pd.notna(avg_weekly_demand) else 0.0
demand_base = df_fc[forecast_label].mean() if amazon_drives or avg_weekly_demand == 0 else avg_weekly_demand
base_stock_level = (lead_time_weeks + float(woc_target)) * demand_base

on_hand_begin, po_units = [], []
pipeline_receipts = [0] * (len(df_fc) + lead_time_weeks + 5)
on_hand = int(start_on_hand)

for t in range(len(df_fc)):
    demand_t = int(df_fc.iloc[t][forecast_label])
    arriving = int(pipeline_receipts[t]) if t < len(pipeline_receipts) else 0
    on_hand += arriving
    open_pos = sum(pipeline_receipts[t+1 : t + lead_time_weeks + 1]) if lead_time_weeks > 0 else 0
    inventory_position = on_hand + open_pos
    order_qty = max(int(round(base_stock_level - inventory_position)), 0)
    if lead_time_weeks > 0 and t + lead_time_weeks < len(pipeline_receipts):
        pipeline_receipts[t + lead_time_weeks] += order_qty
    elif lead_time_weeks == 0:
        on_hand += order_qty
    on_hand_begin.append(int(on_hand))
    po_units.append(int(order_qty))
    on_hand = max(on_hand - demand_t, 0)

df_fc["On_Hand_Begin"] = on_hand_begin
df_fc["Predicted_PO_Units"] = po_units
df_fc["Predicted_SellIn_$"] = (df_fc["Predicted_PO_Units"] * float(unit_price)).round(2)

df_fc["Weeks_Of_Cover"] = np.where(
    df_fc[forecast_label] > 0,
    (df_fc["On_Hand_Begin"] / df_fc[forecast_label]).round(2),
    np.nan
)
df_fc["Date"] = pd.to_datetime(df_fc["Week_Start"]).dt.strftime("%d-%m-%Y")

# ------------------------ Plot ------------------------
st.subheader(f"{len(df_fc)}-Week Forecast & Predicted POs")
if projection_type == "Sales $":
    primary_key, primary_title = "Projected_Sales", "Sales $"
    secondary_key, secondary_title = forecast_label, "Units"
else:
    primary_key, primary_title = forecast_label, "Units"
    secondary_key, secondary_title = "Projected_Sales", "Sales $"

if PLOTLY_INSTALLED:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc[primary_key],
                             name=f"{primary_title} ({'Projected' if primary_key=='Projected_Sales' else 'Sell-out Forecast'})",
                             yaxis="y", line=dict(color=AMZ_ORANGE)))
    fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc["Predicted_PO_Units"],
                             name="Predicted PO Units (Sell-in)",
                             yaxis="y" if primary_key == forecast_label else "y2",
                             line=dict(color=AMZ_BLUE)))
    if secondary_key != primary_key:
        fig.add_trace(go.Scatter(x=df_fc["Week_Start"], y=df_fc[secondary_key],
                                 name=f"{secondary_title} ({'Projected' if secondary_key=='Projected_Sales' else 'Sell-out Forecast'})",
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
    st.line_chart(df_fc.set_index("Week_Start")[[forecast_label, "Predicted_PO_Units"]])
    st.line_chart(df_fc.set_index("Week_Start")["Projected_Sales"])

# ------------------------ Summary + Detail ------------------------
st.subheader("Summary Metrics")
total_po_units = int(df_fc["Predicted_PO_Units"].sum())
total_sellin = float(df_fc["Predicted_SellIn_$"].sum())
avg_sellin = float(df_fc["Predicted_SellIn_$"].mean())
recap = pd.DataFrame({
    "Metric": ["Total Predicted PO Units", "Total Predicted Sell-In $", "Avg Weekly Sell-In $"],
    "Value": [f"{total_po_units:,}", f"${total_sellin:,.2f}", f"${avg_sellin:,.2f}"]
})
st.table(recap)

st.subheader("Detailed Plan")
st.dataframe(
    df_fc[["Date", forecast_label, "Projected_Sales", "On_Hand_Begin", "Predicted_PO_Units", "Predicted_SellIn_$", "Weeks_Of_Cover"]],
    use_container_width=True
)

st.markdown(
    f"<div style='text-align:center;color:gray;margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True,
)
