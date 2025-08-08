"""
Amazon Sell-In Forecast Streamlit App (Amazon Branded)

Dependencies (add to requirements.txt):

streamlit>=1.0
pandas>=1.3
numpy>=1.21
regex
# At least one forecasting engine
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
    page_title="Amazon Sell-In Forecast",
    page_icon=AMZ_LOGO,
    layout="wide"
)
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Sell-In Forecast</h1>"
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

# WOC slider and horizon
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12)

st.markdown("---")

# Run forecast trigger
run_button = st.button("Run Forecast")
if not run_button:
    st.info("Click 'Run Forecast' to run the sell-in forecast")
    st.stop()

# Default file paths
DEFAULT_SALES     = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
DEFAULT_INVENTORY = "/mnt/data/Inventory_ASIN_Manufacturing_Retail_UnitedStates_Custom_8-6-2025_8-6-2025.csv"
DEFAULT_UPSTREAM  = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

# Resolve file sources
sales_path = sales_file if sales_file else (DEFAULT_SALES if os.path.exists(DEFAULT_SALES) else None)
inv_path   = inv_file   if inv_file   else (DEFAULT_INVENTORY if os.path.exists(DEFAULT_INVENTORY) else None)
upstream_path = fcst_file if fcst_file else (DEFAULT_UPSTREAM if os.path.exists(DEFAULT_UPSTREAM) else None)

if not sales_path or not inv_path:
    st.error("Sales history and inventory files are required.")
    st.stop()

# Load sales data
# Skip metadata row and read raw data
raw_sales = pd.read_csv(sales_path, skiprows=1)
# Dynamically find the week column (first column)
week_col = raw_sales.columns[0]
# Parse week start date from the first part of the week range
raw_sales['Week_Start'] = pd.to_datetime(
    raw_sales[week_col].astype(str).str.split(' - ').str[0].str.strip(),
    errors='coerce'
)
# Drop rows without valid dates
df_sales = raw_sales.dropna(subset=['Week_Start']).copy()
# Select metric column based on projection type (dynamic)
cols = raw_sales.columns.tolist()
if projection_type == 'Units':
    # find first column name containing 'unit' (case-insensitive)
    unit_cols = [c for c in cols if re.search(r'unit', c, re.IGNORECASE)]
    if not unit_cols:
        st.error("No 'Units' column found in sales data.")
        st.stop()
    y_col = unit_cols[0]
    forecast_label = 'Sell-Out Units'
    y_label = 'Units'
else:
    # find 'sales' column
    sales_cols = [c for c in cols if re.search(r'sales', c, re.IGNORECASE)]
    if sales_cols:
        y_col = sales_cols[0]
        forecast_label = 'Sell-Out Sales'
        y_label = 'Sales $'
    else:
        # fallback to units
        unit_cols = [c for c in cols if re.search(r'unit', c, re.IGNORECASE)]
        if not unit_cols:
            st.error("No 'Sales' or 'Units' column found in sales data.")
            st.stop()
        y_col = unit_cols[0]
        forecast_label = 'Sell-Out Units'
        y_label = 'Units'
# Clean and cast projection metric to numeric with coercion
numeric_series = df_sales[y_col].astype(str).str.replace('[^0-9.]', '', regex=True)
# Convert to numeric, set invalid parsing as zero
df_sales['y'] = pd.to_numeric(numeric_series, errors='coerce').fillna(0)
# Prepare final sales DataFrame
hist = df_sales[['Week_Start','y']]

# Historical data cutoff
today = pd.to_datetime(datetime.now().date())
hist = df_sales[df_sales['Week_Start'] <= today]
if hist.empty:
    st.error("No historical data before or on today’s date.")
    st.stop()
last_hist = hist['Week_Start'].max()
start_fcst = max(last_hist + timedelta(weeks=1), today + timedelta(weeks=1))
future_dates = pd.date_range(start=start_fcst, periods=periods, freq='W')

# Forecast implementation
if model_choice == 'Prophet':
    model = Prophet(weekly_seasonality=True)
    model.fit(hist.rename(columns={'Week_Start':'ds'}))
    future = pd.DataFrame({'ds': future_dates})
    fcst_df = model.predict(future)[['ds','yhat']].rename(columns={'ds':'Week_Start'})
elif model_choice == 'ARIMA':
    hw = hist.set_index('Week_Start').resample('W-SUN').sum().reset_index()
    ar_model = ARIMA(hw['y'], order=(1,1,1)).fit()
    preds = ar_model.get_forecast(steps=periods)
    fcst_df = pd.DataFrame({'Week_Start': future_dates, 'yhat': preds.predicted_mean.values})
else:
    last_val = hist['y'].iloc[-1]
    fcst_df = pd.DataFrame({'Week_Start': future_dates, 'yhat': last_val})
# Rename forecast column to Sell-Out label
fcst_df = fcst_df.rename(columns={'yhat': forecast_label})

# Load initial inventory
df_inv = pd.read_csv(inv_path, skiprows=1)
oh_raw = df_inv['Sellable On Hand Units'].iloc[0]
try:
    init_inv = int(str(oh_raw).replace(',',''))
except:
    init_inv = float(str(oh_raw).replace(',',''))

# Compute dynamic inventory and sell-in to maintain constant WOC
desired_inv = init_inv
inv_list = []
sellin_list = []
prev_inv = init_inv
for _, row in fcst_df.iterrows():
    demand = row[forecast_label]
    # desired ending inventory to hit WOC target
    desired_inv = demand * woc_target
    # sell-in needed this week to reach desired inventory
    sell_in = desired_inv - (prev_inv - demand)
    # record values
    inv_list.append(int(round(desired_inv)))
    sellin_list.append(int(round(sell_in)))
    # update for next iteration
    prev_inv = desired_inv

# Build result DataFrame
result = fcst_df.copy()
result['Inventory_On_Hand'] = inv_list
result['Sell_In_Forecast'] = sellin_list
# Weeks of Cover now constant by design
result['Weeks_Of_Cover'] = woc_target

# Load and merge Amazon sell-out forecast
if upstream_path:
    df_up = pd.read_csv(upstream_path, skiprows=1)
    rec = []
    for col in df_up.columns:
        if col.startswith('Week '):
            m = re.search(r'\((\d{1,2} [A-Za-z]+)', col)
            if not m:
                continue
            date_str = m.group(1) + ' ' + str(datetime.now().year)
            try:
                ds = pd.to_datetime(date_str, format='%d %b %Y')
            except:
                continue
            val_str = str(df_up[col].iloc[0]).replace(',','')
            try:
                val = round(float(val_str))
            except:
                continue
            rec.append({'Week_Start': ds, 'Amazon_Sellout_Forecast': val})
    if rec:
        upstream_df = pd.DataFrame(rec)
        result = result.merge(upstream_df, on='Week_Start', how='left')

# Format date for display
result['Formatted_Week_Start'] = result['Week_Start'].dt.strftime('%d-%m-%Y')

# Display chart including historical performance
st.subheader(f"{periods}-Week Sell-In Forecast ({projection_type})")

# Prepare historical series
hist_series = hist[['Week_Start','y']].copy()
hist_series = hist_series.rename(columns={'y': f'Historical_{forecast_label}'})
# Round historical to integer units
hist_series[f'Historical_{forecast_label}'] = hist_series[f'Historical_{forecast_label}'].round(0).astype(int)

# Merge history and future results for chart
to_merge = [forecast_label, 'Sell_In_Forecast']
if 'Amazon_Sellout_Forecast' in result.columns:
    to_merge.insert(1, 'Amazon_Sellout_Forecast')
chart_df = pd.merge(
    hist_series,
    result[['Week_Start'] + to_merge],
    on='Week_Start', how='outer'
).sort_values('Week_Start')

metrics = [f'Historical_{forecast_label}'] + to_merge

if PLOTLY_INSTALLED:
    fig = px.line(
        chart_df, x='Week_Start', y=metrics,
        labels={'value': y_label, 'variable':'Metric'},
        title="Historical vs Forecast vs Sell-In"
    )
    fig.update_layout(legend_title_text='')
    fig.update_xaxes(tickformat="%d-%m-%Y")
    # style lines
    fig.update_traces(selector=dict(name=f'Historical_{forecast_label}'), line=dict(color=AMZ_BLUE, dash='solid'))
    if 'Amazon_Sellout_Forecast' in chart_df.columns:
        fig.update_traces(selector=dict(name='Amazon_Sellout_Forecast'), line=dict(color=AMZ_BLUE, dash='dash'))
    fig.update_traces(selector=dict(name=forecast_label), line=dict(color=AMZ_ORANGE, dash='dot'))
    fig.update_traces(selector=dict(name='Sell_In_Forecast'), line=dict(color=AMZ_ORANGE, dash='solid'))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(chart_df.set_index('Week_Start')[metrics])

# Display table
base_cols = ['Formatted_Week_Start', forecast_label, 'Sell_In_Forecast', 'Inventory_On_Hand', 'Weeks_Of_Cover']
if 'Amazon_Sellout_Forecast' in result.columns:
    base_cols.insert(2, 'Amazon_Sellout_Forecast')
st.dataframe(result[base_cols])

# Footer
st.markdown(
    f"<div style='text-align:center; color:gray; margin-top:20px;'>© {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
