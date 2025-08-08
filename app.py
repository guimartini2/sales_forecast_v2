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

# Branding
AMAZON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZN_ORANGE = "#FF9900"
AMZN_BLUE = "#146EB4"

# Page config
st.set_page_config(page_title="Amazon Sell-In Forecast", page_icon=AMAZON_LOGO_URL, layout="wide")
st.markdown(
    f"<div style='display:flex;align-items:center;'>"
    f"<img src='{AMAZON_LOGO_URL}' width=100>"
    f"<h1 style='margin-left:10px;color:{AMZN_BLUE};'>Amazon Sell-In Forecast</h1>"
    f"</div>", unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
inv_file   = st.sidebar.file_uploader("Inventory snapshot CSV", type=["csv"])
fcst_file  = st.sidebar.file_uploader("Upstream forecast CSV", type=["csv"])
projection_type = st.sidebar.selectbox("Projection Type", ["Units","Sales $"])

# Model options
model_opts = []
if PROPHET_INSTALLED: model_opts.append("Prophet")
if ARIMA_INSTALLED:   model_opts.append("ARIMA")
if XGB_INSTALLED:     model_opts.append("XGBoost")
if not model_opts:
    st.error("Install at least one forecasting library: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Model", model_opts)
woc_target   = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods      = st.sidebar.number_input("Forecast Horizon (weeks)", 4, 52, 12)

st.markdown("---")

# Default file paths
default_sales     = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_inventory = "/mnt/data/Inventory_ASIN_Manufacturing_Retail_UnitedStates_Custom_8-6-2025_8-6-2025.csv"
default_fcst      = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

sales_path = sales_file if sales_file else (default_sales if os.path.exists(default_sales) else None)
inv_path   = inv_file   if inv_file   else (default_inventory if os.path.exists(default_inventory) else None)
fcst_path  = fcst_file  if fcst_file  else (default_fcst if os.path.exists(default_fcst) else None)
if not sales_path or not inv_path:
    st.error("Sales history and inventory files are required.")
    st.stop()

# Load sales data
df_sales = pd.read_csv(sales_path, skiprows=1)
# Extract date
df_sales['Week_Start'] = pd.to_datetime(df_sales['Week'].str.split(' - ').str[0].str.strip())
# Parse metric
if projection_type == 'Units':
    df_sales['y'] = df_sales['Ordered Units'].str.replace(',','').astype(int)
    y_label = 'Units'
else:
    if 'Ordered Sales' in df_sales.columns:
        df_sales['y'] = df_sales['Ordered Sales'].str.replace('[^0-9.]','', regex=True).astype(float)
        y_label = 'Sales $'
    else:
        df_sales['y'] = df_sales['Ordered Units'].str.replace(',','').astype(int)
        y_label = 'Units'
df_sales = df_sales[['Week_Start','y']]

# Filter to historical up to today
today = pd.to_datetime(datetime.now().date())
hist = df_sales[df_sales['Week_Start'] <= today]
if hist.empty:
    st.error("No historical data up to today.")
    st.stop()
last_hist = hist['Week_Start'].max()
# Forecast dates start next week
start_forecast = max(last_hist + timedelta(weeks=1), today + timedelta(weeks=1))
dates = pd.date_range(start=start_forecast, periods=periods, freq='W')

# Run forecast
if model_choice == 'Prophet':
    m = Prophet(weekly_seasonality=True)
    m.fit(hist.rename(columns={'Week_Start':'ds'}))
    future = pd.DataFrame({'ds': dates})
    fc = m.predict(future)[['ds','yhat']].rename(columns={'ds':'Week_Start'})
elif model_choice == 'ARIMA':
    hw = hist.set_index('Week_Start').resample('W-SUN').sum().reset_index()
    ar = ARIMA(hw['y'], order=(1,1,1)).fit()
    pred = ar.get_forecast(steps=periods)
    fc = pd.DataFrame({'Week_Start':dates, 'yhat':pred.predicted_mean.values})
else:
    last_val = hist['y'].iloc[-1]
    fc = pd.DataFrame({'Week_Start':dates, 'yhat':last_val})

# Load inventory
df_inv = pd.read_csv(inv_path, skiprows=1)
oh_val = int(str(df_inv['Sellable On Hand Units'].iloc[0]).replace(',',''))
inv_df = pd.DataFrame({'Week_Start':dates, 'Inventory_On_Hand':oh_val})

# Combine
result = fc.merge(inv_df, on='Week_Start')
result['Weeks_Of_Cover'] = result['Inventory_On_Hand'] / result['yhat']
result['Sell_In_Forecast'] = result['Inventory_On_Hand'] / woc_target

# Upstream merge
if fcst_path:
    df_u = pd.read_csv(fcst_path, skiprows=1)
    rec = []
    for col in df_u.columns:
        if col.startswith('Week '):
            m = re.search(r'\(([^)]+)\)', col)
            if m:
                d = pd.to_datetime(m.group(1)+' 2025', format='%d %b %Y')
                val = float(df_u[col].iloc[0].replace(',',''))
                rec.append({'Week_Start':d, 'Upstream_Forecast':val})
    upstream = pd.DataFrame(rec)
    result = result.merge(upstream, on='Week_Start', how='left')

# Rename forecast column
y_col = f'Forecasted_{y_label}'
result = result.rename(columns={'yhat':y_col})

# Display chart
st.subheader(f"{periods}-Week Sell-In Forecast ({projection_type})")
metrics = [y_col, 'Inventory_On_Hand', 'Sell_In_Forecast'] + (['Upstream_Forecast'] if 'Upstream_Forecast' in result else [])
if PLOTLY_INSTALLED:
    fig = px.line(result, x='Week_Start', y=metrics, labels={'value':y_label,'variable':'Metric'})
    fig.update_layout(legend_title_text='')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(result.set_index('Week_Start')[metrics])

# Display table
display_cols = ['Week_Start', y_col, 'Sell_In_Forecast', 'Inventory_On_Hand', 'Weeks_Of_Cover']
if 'Upstream_Forecast' in result: display_cols.insert(3,'Upstream_Forecast')
st.dataframe(result[display_cols].round(2))

# Footer
st.markdown(
    f"<div style='text-align:center;color:gray;margin-top:20px;'>© {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
