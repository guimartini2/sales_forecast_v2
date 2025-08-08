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
fcst_file  = st.sidebar.file_uploader("Upstream forecast CSV", type=["csv"])
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
# Default file paths
DEFAULT_SALES     = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
DEFAULT_INVENTORY = "/mnt/data/Inventory_ASIN_Manufacturing_Retail_UnitedStates_Custom_8-6-2025_8-6-2025.csv"
DEFAULT_FORECAST  = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"

# Resolve file sources
sales_path = sales_file if sales_file else (DEFAULT_SALES if os.path.exists(DEFAULT_SALES) else None)
inv_path   = inv_file   if inv_file   else (DEFAULT_INVENTORY if os.path.exists(DEFAULT_INVENTORY) else None)
fcst_path  = fcst_file  if fcst_file  else (DEFAULT_FORECAST if os.path.exists(DEFAULT_FORECAST) else None)

if not sales_path or not inv_path:
    st.error("Sales history and inventory files are required.")
    st.stop()

# Load sales data
df_sales = pd.read_csv(sales_path, skiprows=1)
# Parse week start date
df_sales['Week_Start'] = pd.to_datetime(
    df_sales['Week'].str.split(' - ').str[0].str.strip()
)
# Select metric
if projection_type == 'Units':
    df_sales['y'] = df_sales['Ordered Units'].str.replace(',','').astype(int)
    y_label = 'Units'
else:
    if 'Ordered Sales' in df_sales.columns:
        df_sales['y'] = df_sales['Ordered Sales']\
            .str.replace('[^0-9.]','', regex=True)\
            .astype(float)
        y_label = 'Sales $'
    else:
        df_sales['y'] = df_sales['Ordered Units'].str.replace(',','').astype(int)
        y_label = 'Units'
# Filter to needed columns
df_sales = df_sales[['Week_Start','y']]

# Historical data up to today
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
    fcst_df = pd.DataFrame({
        'Week_Start': future_dates,
        'yhat': preds.predicted_mean.values
    })
else:  # XGBoost fallback: last value
    last_val = hist['y'].iloc[-1]
    fcst_df = pd.DataFrame({
        'Week_Start': future_dates,
        'yhat': last_val
    })
# Rename forecast column
forecast_col = f'Forecasted_{y_label}'
fcst_df = fcst_df.rename(columns={'yhat': forecast_col})

# Load inventory snapshot
df_inv = pd.read_csv(inv_path, skiprows=1)
oh_raw = df_inv['Sellable On Hand Units'].iloc[0]
try:
    on_hand = int(str(oh_raw).replace(',',''))
except:
    on_hand = float(str(oh_raw).replace(',',''))
inv_df = pd.DataFrame({
    'Week_Start': future_dates,
    'Inventory_On_Hand': on_hand
})

# Combine results
result = fcst_df.merge(inv_df, on='Week_Start')
# Round to whole units
numeric_cols = [forecast_col, 'Inventory_On_Hand']
if 'Upstream_Forecast' in result.columns:
    numeric_cols.append('Upstream_Forecast')
# Weeks_Of_Cover may be fractional, round to 2 decimal
result['Weeks_Of_Cover'] = (result['Inventory_On_Hand'] / result[forecast_col]).round(2)
# Round forecasts and inventory to integers
result[forecast_col] = result[forecast_col].round(0).astype(int)
result['Inventory_On_Hand'] = result['Inventory_On_Hand'].round(0).astype(int)
# Sell-in forecast as integer
result['Sell_In_Forecast'] = (result['Inventory_On_Hand'] / woc_target).round(0).astype(int)

# Upstream forecast merge
if fcst_path:
    df_u = pd.read_csv(fcst_path, skiprows=1)
    rec = []
    for col in df_u.columns:
        if col.startswith('Week '):
            m = re.search(r'\((\d{1,2} [A-Za-z]+)', col)
            if not m:
                continue
            date_str = m.group(1) + ' ' + str(datetime.now().year)
            try:
                ds = pd.to_datetime(date_str, format='%d %b %Y')
            except:
                continue
            val_str = str(df_u[col].iloc[0]).replace(',','')
            try:
                val = float(val_str)
            except:
                continue
            rec.append({'Week_Start': ds, 'Upstream_Forecast': round(val)})
    if rec:
        upstream_df = pd.DataFrame(rec)
        result = result.merge(upstream_df, on='Week_Start', how='left')

# Format date for display
result['Formatted_Week_Start'] = result['Week_Start'].dt.strftime('%d-%m-%Y')

# Display chart
st.subheader(f"{periods}-Week Sell-In Forecast ({projection_type})")
metrics = [forecast_col, 'Inventory_On_Hand', 'Sell_In_Forecast']
if 'Upstream_Forecast' in result.columns:
    metrics.insert(2, 'Upstream_Forecast')
if PLOTLY_INSTALLED:
    fig = px.line(
        result, x='Week_Start', y=metrics,
        labels={'value': y_label, 'variable': 'Metric'},
        title="Sell-In vs Demand vs Inventory"
    )
    fig.update_layout(legend_title_text='')
    fig.update_xaxes(tickformat="%d-%m-%Y")
    fig.update_traces(
        selector=dict(name=forecast_col),
        line=dict(color=AMZ_ORANGE, dash='dash')
    )
    fig.update_traces(
        selector=dict(name='Inventory_On_Hand'),
        line=dict(color=AMZ_BLUE)
    )
    fig.update_traces(
        selector=dict(name='Sell_In_Forecast'),
        line=dict(color=AMZ_ORANGE)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    basic_df = result.set_index('Week_Start')[metrics]
    st.line_chart(basic_df)

# Display table
display_cols = ['Formatted_Week_Start', forecast_col, 'Sell_In_Forecast', 'Inventory_On_Hand', 'Weeks_Of_Cover']
if 'Upstream_Forecast' in result.columns:
    display_cols.insert(3, 'Upstream_Forecast')
st.dataframe(result[display_cols])

# Footer
st.markdown(
    f"<div style='text-align:center; color:gray; margin-top:20px;'>© {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
