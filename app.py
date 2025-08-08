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

# Attempt Plotly import
PLOTLY_INSTALLED = False
try:
    import plotly.express as px
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False

# Forecasting libraries availability
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
AMAZON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMAZON_PRIMARY = "#FF9900"
AMAZON_SECONDARY = "#146EB4"

# Streamlit page config
st.set_page_config(page_title="Amazon Sell-In Forecast", page_icon=AMAZON_LOGO_URL, layout="wide")
# Header
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMAZON_LOGO_URL}' width='120'>"
    f"<h1 style='margin-left:10px; color:{AMAZON_SECONDARY};'>Amazon Sell-In Forecast</h1>"
    f"</div>", unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
inv_file = st.sidebar.file_uploader("Inventory snapshot CSV", type=["csv"])
fcst_file = st.sidebar.file_uploader("Upstream forecast CSV", type=["csv"])

projection_type = st.sidebar.selectbox("Projection Type", ["Units", "Sales $"])

# Model selection
model_options = []
if PROPHET_INSTALLED: model_options.append("Prophet")
if ARIMA_INSTALLED:   model_options.append("ARIMA")
if XGB_INSTALLED:     model_options.append("XGBoost")
if not model_options:
    st.error("Install at least one of: prophet, statsmodels, xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Model", model_options)

woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12)

st.markdown("---")

# Load default paths
DEFAULT_SALES = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
DEFAULT_INVENTORY = "/mnt/data/Inventory_ASIN_Manufacturing_Retail_UnitedStates_Custom_8-6-2025_8-6-2025.csv"
DEFAULT_FORECAST = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

# Resolve file sources
sales_path = sales_file if sales_file else (DEFAULT_SALES if os.path.exists(DEFAULT_SALES) else None)
inv_path   = inv_file   if inv_file   else (DEFAULT_INVENTORY if os.path.exists(DEFAULT_INVENTORY) else None)
fcst_path  = fcst_file  if fcst_file  else (DEFAULT_FORECAST if os.path.exists(DEFAULT_FORECAST) else None)
if not sales_path or not inv_path:
    st.error("Sales history and inventory files are required.")
    st.stop()

# Read sales
df_sales = pd.read_csv(sales_path, skiprows=1)
# Parse date and metrics
df_sales['Week_Start'] = pd.to_datetime(df_sales['Week'].str.split(' - ').str[0].str.strip())
if projection_type == "Units":
    df_sales['y'] = df_sales['Ordered Units'].str.replace(',','').astype(int)
    y_label = "Units"
else:
    # Attempt to parse sales dollars
    if 'Ordered Sales' in df_sales.columns:
        df_sales['y'] = df_sales['Ordered Sales']\
            .str.replace('[^0-9.]','', regex=True)\
            .astype(float)
        y_label = "Sales $"
    else:
        st.sidebar.warning("'Ordered Sales' column not found; defaulting to Units.")
        df_sales['y'] = df_sales['Ordered Units'].str.replace(',','').astype(int)
        y_label = "Units"

# Continue with filtering history, forecasting, and merging
# Table

display_cols = ['Week_Start', f'Forecasted_{y_label}', 'Sell_In_Units', 'Inventory_On_Hand', 'Weeks_Of_Cover']
if 'Upstream_Forecast' in result: display_cols.insert(3,'Upstream_Forecast')
st.dataframe(result[display_cols].round(2))

# Footer
st.markdown(
    f"<div style='text-align:center; color:gray; margin-top:20px;'>© {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
