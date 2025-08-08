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
import plotly.express as px

# Amazon branding
AMAZON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMAZON_PRIMARY = "#FF9900"  # Amazon orange
AMAZON_SECONDARY = "#146EB4"  # Amazon blue

# Page config
st.set_page_config(
    page_title="Amazon Sell-In Forecast", layout="wide",
    page_icon=AMAZON_LOGO_URL
)
# Header with logo
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMAZON_LOGO_URL}' width='120'>"
    f"<h1 style='margin-left:10px; color:{AMAZON_SECONDARY};'>Amazon Sell-In Forecast</h1>"
    f"</div>", unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Data Inputs & Settings")
# File upload
sales_file = st.sidebar.file_uploader("Sales history CSV (Units/Sales)", type=["csv"])
inv_file = st.sidebar.file_uploader("Inventory snapshot CSV", type=["csv"])
fcst_file = st.sidebar.file_uploader("Upstream forecast CSV", type=["csv"])

# Projection type
projection_type = st.sidebar.selectbox(
    "Projection Type", ["Units", "Sales $"]
)

# Forecast parameters
model_choice = st.sidebar.selectbox(
    "Model", [opt for opt in ("Prophet","ARIMA","XGBoost") if opt in globals().get(f"{opt.upper()}_INSTALLED", [True])]
)
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = st.sidebar.number_input("Forecast Horizon (weeks)", min_value=4, max_value=52, value=12)

st.markdown("---")

# Load data (uploads not shown for brevity, similar to default logic)
# ... [assume sales_path, inv_path, fcst_path logic unchanged] ...

# After loading df_sales and df_inv as before
# Parse ds and y
if projection_type == "Units":
    df_sales['y'] = df_sales['Ordered Units'].str.replace(',', '').astype(int)
    y_label = "Units"
else:
    if 'Ordered Sales' in df_sales.columns:
        df_sales['y'] = df_sales['Ordered Sales'].str.replace('[^0-9.]','', regex=True).astype(float)
    else:
        st.sidebar.warning("Column 'Ordered Sales' not found; defaulting to Units.")
        df_sales['y'] = df_sales['Ordered Units'].str.replace(',', '').astype(int)
        y_label = "Units"
    y_label = "Sales $"

# Ensure history is up to today
today = pd.to_datetime(datetime.now().date())
hist = hist[hist['ds'] <= today]
last_hist_date = hist['ds'].max()
start_date = max(last_hist_date + timedelta(weeks=1), today + timedelta(weeks=1))

# Forecasting (use start_date)
future_idx = pd.date_range(start=start_date, periods=periods, freq='W')
if model_choice == 'Prophet':
    m = Prophet(weekly_seasonality=True)
    m.fit(hist)
    future = pd.DataFrame({'ds': future_idx})
    forecast = m.predict(future)[['ds','yhat']]
elif model_choice == 'ARIMA':
    hist_w = hist.set_index('ds').resample('W-SUN').sum().reset_index()
    arima_res = ARIMA(hist_w['y'], order=(1,1,1)).fit()
    f = arima_res.get_forecast(steps=periods)
    forecast = pd.DataFrame({'ds': future_idx, 'yhat': f.predicted_mean.values})
else:
    last = hist['y'].iloc[-1]
    forecast = pd.DataFrame({'ds': future_idx, 'yhat': last})

# Inventory and WOC
oh = int(str(df_inv['Sellable On Hand Units'].iloc[0]).replace(',',''))
inv_df = pd.DataFrame({'ds': future_idx, 'Inventory_On_Hand': oh})
result = forecast.merge(inv_df, on='ds')
result['Weeks_Of_Cover'] = result['Inventory_On_Hand'] / result['yhat']
result['Sell_In_Units'] = result['Inventory_On_Hand'] / woc_target

# Merge upstream
if fcst_path:
    result = result.merge(upstream, on='ds', how='left')

# Rename and format
result = result.rename(columns={'ds':'Week_Start', 'yhat':f'Forecasted_{y_label}'})

# Display
st.subheader(f"{periods}-Week Sell-In Forecast ({projection_type})")

# Plot with Amazon colors
fig = px.line(
    result, x='Week_Start', y=[f'Forecasted_{y_label}','Inventory_On_Hand','Sell_In_Units'] + (["Upstream_Forecast"] if 'Upstream_Forecast' in result.columns else []),
    labels={"value":y_label, "variable":"Metric"},
    title="Sell-In vs Demand vs Inventory",
)
fig.update_traces(selector=dict(name=f'Forecasted_{y_label}'), line=dict(color=AMAZON_PRIMARY, dash='dash'))
fig.update_traces(selector=dict(name='Inventory_On_Hand'), line=dict(color=AMAZON_SECONDARY))
fig.update_traces(selector=dict(name='Sell_In_Units'), line=dict(color=AMAZON_PRIMARY))
st.plotly_chart(fig, use_container_width=True)

# Data table
st.dataframe(
    result[['Week_Start','Forecasted_'+y_label,'Sell_In_Units','Inventory_On_Hand','Weeks_Of_Cover']+
           (["Upstream_Forecast"] if 'Upstream_Forecast' in result.columns else [])]
    .round(2)
)

# Footer
st.markdown(
    "<div style='text-align:center; margin-top:20px; color:gray;'>" +
    f"&copy; {datetime.now().year} Amazon Internal Tool" +
    "</div>", unsafe_allow_html=True
)
