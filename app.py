"""
Amazon Sell-In Forecast Streamlit App

Dependencies (add to requirements.txt):

streamlit>=1.0
pandas>=1.3
numpy>=1.21
regex
# At least one forecasting engine
prophet>=1.0    # or fbprophet>=0.7
statsmodels>=0.13
xgboost>=1.7
"""

import os
import re
import streamlit as st
import pandas as pd
import numpy as np

# Default file paths
DEFAULT_SALES = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
DEFAULT_INVENTORY = "/mnt/data/Inventory_ASIN_Manufacturing_Retail_UnitedStates_Custom_8-6-2025_8-6-2025.csv"
DEFAULT_FORECAST = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

# Check forecasting libraries
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

# Streamlit setup
st.set_page_config(page_title="Amazon Sell-In Forecast", layout="wide")
st.title("Amazon Sell-In Forecast by ASIN")

# Sidebar: file inputs or defaults
st.sidebar.header("Data Inputs (optional override)")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
inv_file = st.sidebar.file_uploader("Inventory snapshot CSV", type=["csv"])
fcst_file = st.sidebar.file_uploader("Upstream forecast CSV", type=["csv"])

# Use defaults if not uploaded
sales_path = None
if sales_file is not None:
    sales_path = sales_file
elif os.path.exists(DEFAULT_SALES):
    sales_path = DEFAULT_SALES
    st.sidebar.markdown(f"Using default sales file: `{DEFAULT_SALES}`")
else:
    st.sidebar.error("Sales history file not provided or found.")

inv_path = None
if inv_file is not None:
    inv_path = inv_file
elif os.path.exists(DEFAULT_INVENTORY):
    inv_path = DEFAULT_INVENTORY
    st.sidebar.markdown(f"Using default inventory file: `{DEFAULT_INVENTORY}`")
else:
    st.sidebar.error("Inventory file not provided or found.")

fcst_path = None
if fcst_file is not None:
    fcst_path = fcst_file
elif os.path.exists(DEFAULT_FORECAST):
    fcst_path = DEFAULT_FORECAST
    st.sidebar.markdown(f"Using default forecast file: `{DEFAULT_FORECAST}`")

# Sidebar: parameters
st.sidebar.header("Forecast Parameters")
model_options = []
if PROPHET_INSTALLED:
    model_options.append("Prophet")
if ARIMA_INSTALLED:
    model_options.append("ARIMA")
if XGB_INSTALLED:
    model_options.append("XGBoost")
if not model_options:
    st.error("Install at least one forecasting library: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Model", model_options)
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)

if st.sidebar.button("Run Forecast"):
    # Read sales history
    df_sales = pd.read_csv(sales_path, skiprows=1)
    # parse ds and y
    df_sales['ds'] = pd.to_datetime(df_sales['Week'].str.split(' - ').str[0].str.strip())
    df_sales['y'] = df_sales['Ordered Units'].str.replace(',', '').astype(int)
    hist = df_sales[['ds','y']]

    # Read inventory
    df_inv = pd.read_csv(inv_path, skiprows=1)
    # parse Sellable On Hand Units as numeric
    oh_raw = df_inv['Sellable On Hand Units'].iloc[0]
    try:
        oh = int(str(oh_raw).replace(',',''))
    except:
        oh = float(str(oh_raw).replace(',',''))

    # Read upstream forecast if available
    upstream = None
    if fcst_path:
        df_fc = pd.read_csv(fcst_path, skiprows=1)
        week_cols = [c for c in df_fc.columns if c.startswith('Week ')]
        rec = []
        for col in week_cols:
            m = re.search(r'Week \d+ \((\d+ \w+)', col)
            if not m: continue
            start = m.group(1) + ' 2025'
            ds = pd.to_datetime(start, format='%d %b %Y')
            val = df_fc[col].iloc[0].replace(',','')
            yhat = float(val)
            rec.append({'ds': ds, 'yhat_up': yhat})
        upstream = pd.DataFrame(rec)

    # Forecasting
    periods = 12
    if model_choice == 'Prophet':
        m = Prophet(weekly_seasonality=True)
        m.fit(hist)
        future = m.make_future_dataframe(periods=periods, freq='W')
        forecast = m.predict(future)[['ds','yhat']].tail(periods)
    elif model_choice == 'ARIMA':
        hist_w = hist.set_index('ds').resample('W-SUN').sum().reset_index()
        model = ARIMA(hist_w['y'], order=(1,1,1)).fit()
        f = model.get_forecast(steps=periods)
        idx = pd.date_range(start=hist_w['ds'].max()+pd.Timedelta(weeks=1), periods=periods, freq='W-SUN')
        forecast = pd.DataFrame({'ds': idx, 'yhat': f.predicted_mean.values})
    else:
        last = hist['y'].iloc[-1]
        idx = pd.date_range(start=hist['ds'].max()+pd.Timedelta(weeks=1), periods=periods, freq='W')
        forecast = pd.DataFrame({'ds': idx, 'yhat': last})

    # Build inventory series and compute WOC
    inv_df = pd.DataFrame({'ds': forecast['ds'], 'on_hand': oh})
    result = forecast.merge(inv_df, on='ds')
    result['woc'] = result['on_hand'] / result['yhat']
    if upstream is not None:
        result = result.merge(upstream, on='ds', how='left')

    # Display results
    st.subheader("Forecast Results")
    cols = ['yhat','on_hand'] + (['yhat_up'] if upstream is not None else [])
    st.line_chart(result.set_index('ds')[cols])
    st.dataframe(result.round(2))
