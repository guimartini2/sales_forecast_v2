"""
Amazon Replenishment Forecast Streamlit App (Amazon Branded)

Key improvements:
- Price input for projected sales
- Dual-axis chart (units vs. dollars)
- Recap summary table
- Overwrite forecast units with Amazon sell-out data

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

# Optional forecasting libraries
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
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    pass

# Amazon branding
AMZ_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
AMZ_ORANGE = "#FF9900"
AMZ_BLUE = "#146EB4"

# Page setup
st.set_page_config(page_title="Amazon Replenishment Forecast", page_icon=AMZ_LOGO, layout="wide")
st.markdown(
    f"<div style='display:flex; align-items:center;'>"
    f"<img src='{AMZ_LOGO}' width=100>"
    f"<h1 style='margin-left:10px; color:{AMZ_BLUE};'>Amazon Replenishment Forecast</h1>"
    f"</div>", unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Data Inputs & Settings")
sales_file = st.sidebar.file_uploader("Sales history CSV", type=["csv"])
fcst_file = st.sidebar.file_uploader("Amazon Sell-Out Forecast CSV", type=["csv"])
projection_type = st.sidebar.selectbox("Projection Type", ["Units", "Sales $"])
init_inv = st.sidebar.number_input("Current On-Hand Inventory (units)", min_value=0, value=26730, step=1)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, value=10.0, step=0.01)

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
    st.info("Click 'Run Forecast' to generate the forecast and replenishment plan.")
    st.stop()

# Load data paths
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_up = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
sales_path = sales_file if sales_file else (default_sales if os.path.exists(default_sales) else None)
up_path = fcst_file if fcst_file else (default_up if os.path.exists(default_up) else None)
if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

# Load and clean sales history
df_raw = pd.read_csv(sales_path, skiprows=1)
df_raw['Week_Start'] = pd.to_datetime(
    df_raw.iloc[:,0].astype(str).str.split(' - ').str[0].str.strip(), errors='coerce'
)
df_raw.dropna(subset=['Week_Start'], inplace=True)

# Determine metric column
y_col = next((c for c in df_raw.columns if re.search(r'(unit|qty|quantity)', c, re.IGNORECASE)), df_raw.columns[1])
forecast_label = 'Forecast_Units'
y_label = 'Units'

# Historical series
df_hist = pd.DataFrame({'Week_Start': df_raw['Week_Start'], 
                        'y': pd.to_numeric(df_raw[y_col].astype(str)
                                            .str.replace('[^0-9.]','',regex=True), errors='coerce').fillna(0)})
df_hist = df_hist[df_hist['Week_Start'] <= pd.to_datetime(datetime.now().date())]
if df_hist.empty:
    st.error("No valid historical data.")
    st.stop()

# Forecast dates
last_date = df_hist['Week_Start'].max()
start_fc = max(
    last_date + timedelta(weeks=1),
    pd.to_datetime(datetime.now().date()) + timedelta(weeks=1)
)
future_idx = pd.date_range(start=start_fc, periods=periods, freq='W')

# Generate forecast
if model_choice=='Prophet':
    m=Prophet(weekly_seasonality=True)
    m.fit(df_hist.rename(columns={'Week_Start':'ds','y':'y'}))
    fut=pd.DataFrame({'ds':future_idx})
    df_fc=m.predict(fut)[['ds','yhat']].rename(columns={'ds':'Week_Start'})
elif model_choice=='ARIMA':
    tmp=df_hist.set_index('Week_Start').resample('W').sum().reset_index()
    ar=ARIMA(tmp['y'],order=(1,1,1)).fit()
    pr=ar.get_forecast(steps=periods)
    df_fc=pd.DataFrame({'Week_Start':future_idx,'yhat':pr.predicted_mean.values})
else:
    last=df_hist['y'].iloc[-1]
    df_fc=pd.DataFrame({'Week_Start':future_idx,'yhat':last})

# Rename forecast column
df_fc[forecast_label]=df_fc['yhat'].round(0).astype(int)

# Override with Amazon sell-out forecast where available
if up_path:
    try:
        df_up_raw = pd.read_csv(up_path, skiprows=1)
    except pd.errors.EmptyDataError:
        st.warning("⚠️ Amazon sell-out forecast file is empty; skipping upstream merge.")
        df_up_raw = pd.DataFrame()
    except Exception:
        df_up_raw = pd.DataFrame()

    if not df_up_raw.empty:
        rec = []
        for c in df_up_raw.columns:
            m = re.search(r"\((\d{1,2} [A-Za-z]+)\)", c)
            if m:
                dt = pd.to_datetime(
                    f"{m.group(1)} {datetime.now().year}",
                    format="%d %b %Y", errors='coerce'
                )
                val = pd.to_numeric(str(df_up_raw[c].iloc[0]).replace(",", ""), errors='coerce')
                if pd.notna(val):
                    rec.append({"Week_Start": dt, "Amazon_Sellout_Forecast": int(round(val))})
        if rec:
            df_up = pd.DataFrame(rec).sort_values("Week_Start")
            df_fc = df_fc.merge(df_up, on="Week_Start", how="left")
            df_fc[forecast_label] = (
                df_fc["Amazon_Sellout_Forecast"]
                .fillna(df_fc[forecast_label])
                .astype(int)
            )

# Compute projected sales
df_fc['Projected_Sales'] = (df_fc[forecast_label] * unit_price).round(2)

# Replenishment logic
on_hand = []
replen = []
prev = init_inv
for _, r in df_fc.iterrows():
    D = r[forecast_label]
    target = D * woc_target
    Q = max(target - prev, 0)
    on_hand.append(int(prev))
    replen.append(int(Q))
    prev = prev + Q - D

df_fc['On_Hand_Begin'] = on_hand
df_fc['Replenishment'] = replen
df_fc['Weeks_Of_Cover'] = (
    df_fc['On_Hand_Begin'] / df_fc[forecast_label]
).replace([np.inf, -np.inf], np.nan).round(2)

# Format date
df_fc['Date'] = df_fc['Week_Start'].dt.strftime('%d-%m-%Y')

# Plot: dual-axis
st.subheader(f"{periods}-Week Forecast & Replenishment")
if PLOTLY_INSTALLED:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_fc['Week_Start'], y=df_fc[forecast_label], name='Forecast Units', yaxis='y1', line=dict(color=AMZ_ORANGE)))
    fig.add_trace(go.Scatter(x=df_fc['Week_Start'], y=df_fc['Replenishment'], name='Replenishment Units', yaxis='y1', line=dict(color=AMZ_BLUE)))
    fig.add_trace(go.Scatter(x=df_fc['Week_Start'], y=df_fc['Projected_Sales'], name='Projected Sales $', yaxis='y2', line=dict(dash='dot', color='green')))  
    fig.update_layout(
        xaxis=dict(title='Week'),
        yaxis=dict(title='Units'),
        yaxis2=dict(title='Sales $', overlaying='y', side='right'),
        legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'),
        hovermode='x unified'
    )
    fig.update_xaxes(tickformat='%d-%m-%Y')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(df_fc.set_index('Week_Start')[[forecast_label, 'Replenishment']])
    st.line_chart(df_fc.set_index('Week_Start')['Projected_Sales'])

# Recap summary table
st.subheader("Summary Metrics")
total_rep = df_fc['Replenishment'].sum()
total_sales = df_fc['Projected_Sales'].sum()
avg_sales = df_fc['Projected_Sales'].mean()
recap = pd.DataFrame({
    'Metric': ['Total Replenishment Units', 'Total Projected Sales $', 'Avg Weekly Sales $'],
    'Value': [f"{total_rep:,}", f"${total_sales:,.2f}", f"${avg_sales:,.2f}"]
})
st.table(recap)

# Detailed table
st.subheader("Detailed Plan")
st.dataframe(df_fc[['Date', forecast_label, 'Projected_Sales', 'On_Hand_Begin', 'Replenishment', 'Weeks_Of_Cover']], use_container_width=True)

# Footer
st.markdown(f"<div style='text-align:center;color:gray;margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>", unsafe_allow_html=True)
