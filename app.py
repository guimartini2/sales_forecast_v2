"""
Amazon Replenishment Forecast Streamlit App (Amazon Branded)

Dependencies (add to requirements.txt):

streamlit>=1.0
pandas>=1.3
numpy>=1.21
regex
prophet>=1.0    # or fbprophet>=0.7
statsmodels>=0.13
xgboost>=1.7
plotly>=5.0
"""
import os
import re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Optional libraries
PLOTLY_INSTALLED = False
try:
    import plotly.express as px
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
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    pass

# Amazon branding
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# Page config
st.set_page_config(
    page_title="Amazon Replenishment Forecast",
    page_icon=AMZ_LOGO,
    layout="wide"
)

st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>",
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
inv_file   = st.sidebar.file_uploader("Inventory snapshot CSV", type=["csv"])
fcst_file  = st.sidebar.file_uploader("Amazon Sell-Out Forecast CSV", type=["csv"])
projection_type = st.sidebar.selectbox("Projection Type", ["Units", "Sales $"])

# Model selection
model_opts = []
if PROPHET_INSTALLED: model_opts.append("Prophet")
if ARIMA_INSTALLED:   model_opts.append("ARIMA")
if XGB_INSTALLED:     model_opts.append("XGBoost")
if not model_opts:
    st.error("Install at least one forecasting engine: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Forecast Model", model_opts)

# Parameters
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12)

st.markdown("---")
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to generate the replenishment plan.")
    st.stop()

# Default file paths
default_sales     = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_inventory = "/mnt/data/Inventory_ASIN_Manufacturing_Retail_UnitedStates_Custom_8-6-2025_8-6-2025.csv"
default_upstream  = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

sales_path   = sales_file if sales_file else (default_sales if os.path.exists(default_sales) else None)
inv_path     = inv_file   if inv_file   else (default_inventory if os.path.exists(default_inventory) else None)
upstream_path= fcst_file  if fcst_file  else (default_upstream if os.path.exists(default_upstream) else None)

if not sales_path or not inv_path:
    st.error("Sales history and inventory snapshot files are required.")
    st.stop()

# Load sales data
raw = pd.read_csv(sales_path, skiprows=1)
week_col = raw.columns[0]
raw['Week_Start'] = pd.to_datetime(
    raw[week_col].astype(str).str.split(' - ').str[0].str.strip(), errors='coerce'
)
raw.dropna(subset=['Week_Start'], inplace=True)

# Select projection metric dynamically
cols = raw.columns.tolist()
if projection_type == 'Units':
    unit_cols = [c for c in cols if re.search(r'unit', c, re.IGNORECASE)]
    y_col = unit_cols[0] if unit_cols else cols[1]
    forecast_label = 'Sell-Out Units'
    y_label = 'Units'
else:
    sales_cols = [c for c in cols if re.search(r'sales', c, re.IGNORECASE)]
    y_col = sales_cols[0] if sales_cols else cols[1]
    forecast_label = 'Sell-Out Sales'
    y_label = 'Sales $'

# Clean and prepare historical data
df_hist = pd.DataFrame({
    'Week_Start': raw['Week_Start'],
    'y': pd.to_numeric(
        raw[y_col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce'
    ).fillna(0)
})
df_hist = df_hist[df_hist['Week_Start'] <= pd.to_datetime(datetime.now().date())]
if df_hist.empty:
    st.error("No valid historical data.")
    st.stop()

# Forecast date range
last_hist = df_hist['Week_Start'].max()
start_fc = max(last_hist + timedelta(weeks=1), pd.to_datetime(datetime.now().date()) + timedelta(weeks=1))
future_idx = pd.date_range(start=start_fc, periods=periods, freq='W')

# Generate forecast
if model_choice == 'Prophet':
    m = Prophet(weekly_seasonality=True)
    m.fit(df_hist.rename(columns={'Week_Start':'ds'}))
    fut = pd.DataFrame({'ds': future_idx})
    df_fc = m.predict(fut)[['ds','yhat']].rename(columns={'ds':'Week_Start'})
elif model_choice == 'ARIMA':
    tmp = df_hist.set_index('Week_Start').resample('W').sum().reset_index()
    ar = ARIMA(tmp['y'], order=(1,1,1)).fit()
    pr = ar.get_forecast(steps=periods)
    df_fc = pd.DataFrame({'Week_Start': future_idx, 'yhat': pr.predicted_mean.values})
else:
    last_val = df_hist['y'].iloc[-1]
    df_fc = pd.DataFrame({'Week_Start': future_idx, 'yhat': last_val})

# Round and rename forecast
df_fc[forecast_label] = df_fc['yhat'].round(0).astype(int)

# Load inventory snapshot dynamically
df_inv = pd.read_csv(inv_path, skiprows=1)
# Dynamic detection of on-hand column
inv_cols = [c for c in df_inv.columns if re.search(r'on hand|sellable', c, re.IGNORECASE)]
if not inv_cols:
    st.error("No inventory 'On Hand' column found.")
    st.stop()
oh_raw = df_inv[inv_cols[0]].iloc[0]
# Robust parse initial inventory: strip non-digits and convert
raw_str = str(oh_raw)
digits = re.sub(r'[^0-9]', '', raw_str)
init_inv = int(digits) if digits else 0

# Compute dynamic safety stock using coefficient of variation
sigma = df_hist['y'].std()
mean_d = df_hist['y'].mean()
cv = sigma / mean_d if mean_d > 0 else 0
# Safety stock per period scales with forecast and covers variability
if 'Sell-Out Units' in df_fc.columns:
    df_fc['Safety_Stock'] = (df_fc['Sell-Out Units'] * cv * np.sqrt(woc_target)).round(0).astype(int)
else:
    df_fc['Safety_Stock'] = 0

# Replenishment logic
replenishment = []
on_hand_begin = []
prev_on = init_inv
for idx, row in df_fc.iterrows():
    D = row['Sell-Out Units']
    S = row['Safety_Stock']
    # Target inventory = demand * WOC + safety buffer
    target_inv = D * woc_target + S
    # Order quantity = needed to reach target inventory
    Q = max(target_inv - prev_on, 0)
    on_hand_begin.append(int(prev_on))
    replenishment.append(int(Q))
    # Update on-hand for next period (start + receipt - demand)
    prev_on = prev_on + Q - D

# Assign results
df_fc['On_Hand_Begin'] = on_hand_begin
df_fc['Replenishment'] = replenishment
# Weeks of Cover = (on-hand + incoming) / demand
df_fc['Weeks_Of_Cover'] = ((df_fc['On_Hand_Begin'] + df_fc['Replenishment']) / df_fc['Sell-Out Units']).round(2)

# Merge Amazon upstream sell-out forecast
if upstream_path:
    df_up = pd.read_csv(upstream_path, skiprows=1)
    rec = []
    for c in df_up.columns:
        if c.startswith('Week '):
            m = re.search(r'\((\d{1,2} [A-Za-z]+)', c)
            if m:
                ds = pd.to_datetime(m.group(1) + ' ' + str(datetime.now().year), format='%d %b %Y', errors='coerce')
                val = pd.to_numeric(str(df_up[c].iloc[0]).replace(',', ''), errors='coerce')
                rec.append({'Week_Start': ds, 'Amazon_Sellout_Forecast': int(round(val))})
    if rec:
        updf = pd.DataFrame(rec)
        df_fc = df_fc.merge(updf, on='Week_Start', how='left')

# Unify Sell-Out Units metric
df_fc['Sell-Out Units'] = df_fc.get('Amazon_Sellout_Forecast', df_fc[forecast_label])

# Format date for display
df_fc['Date'] = df_fc['Week_Start'].dt.strftime('%d-%m-%Y')

# Plot results
st.subheader(f"{periods}-Week Replenishment Plan ({projection_type})")
metrics = ['Sell-Out Units', 'Safety_Stock', 'Replenishment']
if 'Amazon_Sellout_Forecast' in df_fc.columns:
    metrics.insert(1, 'Amazon_Sellout_Forecast')
metrics = [m for m in metrics if m in df_fc.columns]

if PLOTLY_INSTALLED:
    fig = px.line(
        df_fc, x='Week_Start', y=metrics,
        labels={'value': y_label, 'variable': 'Metric'},
        title='Replenishment vs Demand'
    )
    fig.update_xaxes(tickformat='%d-%m-%Y')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(df_fc.set_index('Week_Start')[metrics])

# Display table
base_cols = ['Date', 'Sell-Out Units', 'Safety_Stock', 'Replenishment', 'On_Hand_Begin', 'Weeks_Of_Cover']
if 'Amazon_Sellout_Forecast' in df_fc.columns:
    base_cols.insert(2, 'Amazon_Sellout_Forecast')
display_cols = [c for c in base_cols if c in df_fc.columns]
st.dataframe(df_fc[display_cols])

# Footer
st.markdown(
    f"<div style='text-align:center; color:gray; margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
