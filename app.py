"""
Amazon Sell-In Forecast Streamlit App

Dependencies (add to requirements.txt):

streamlit>=1.0
pandas>=1.3
numpy>=1.21
# At least one forecasting engine
prophet>=1.0    # or fbprophet>=0.7
statsmodels>=0.13
xgboost>=1.7
"""

import streamlit as st
import pandas as pd
import numpy as np

# Check availability of optional forecasting libraries
PROPHET_INSTALLED = False
try:
    from prophet import Prophet
    PROPHET_INSTALLED = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_INSTALLED = True
    except ImportError:
        PROPHET_INSTALLED = False

ARIMA_INSTALLED = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_INSTALLED = True
except ImportError:
    ARIMA_INSTALLED = False

XGB_INSTALLED = False
try:
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False

# Streamlit page configuration
st.set_page_config(page_title="Amazon Sell-In Forecast", layout="wide")
st.title("Amazon SKU-Level Sell-In Forecast")

# Sidebar: Data uploads
st.sidebar.header("Data Inputs")
history_file = st.sidebar.file_uploader("Upload sell-out history (CSV)", type=["csv"])
inventory_file = st.sidebar.file_uploader("Upload inventory levels (CSV)", type=["csv"])
upstream_file = st.sidebar.file_uploader("Upload existing sell-out forecast (CSV, optional)", type=["csv"])

# Sidebar: Forecast parameters
st.sidebar.header("Forecast Parameters")
model_options = []
if PROPHET_INSTALLED:
    model_options.append("Prophet")
if ARIMA_INSTALLED:
    model_options.append("ARIMA")
if XGB_INSTALLED:
    model_options.append("XGBoost")

if not model_options:
    st.error("No forecasting libraries are installed. Please install at least one of: prophet, statsmodels, xgboost.")
    st.stop()

model_choice = st.sidebar.selectbox("Select forecast model", model_options)
woc_target = st.sidebar.slider("Target Weeks of Cover", 1.0, 12.0, 4.0, step=0.5)
# Collect event impacts as CSV text: date,uplift_pct per line
events_text = st.sidebar.text_area(
    "Event uplifts (one per line, format: YYYY-MM-DD,Uplift% like 2025-11-25,20)",
    height=100,
    key="events_text"
)

if st.sidebar.button("Run Forecast"):("Run Forecast"):
    # Validate library availability
    if model_choice == "Prophet" and not PROPHET_INSTALLED:
        st.error("Prophet library not found. Install with: pip install prophet (or fbprophet).")
        st.stop()
    if model_choice == "ARIMA" and not ARIMA_INSTALLED:
        st.error("statsmodels ARIMA not available. Install with: pip install statsmodels.")
        st.stop()
    if model_choice == "XGBoost" and not XGB_INSTALLED:
        st.error("XGBoost not available. Install with: pip install xgboost.")
        st.stop()

    # Validate required uploads
    if not history_file or not inventory_file:
        st.error("Please upload both sell-out history and inventory files to proceed.")
        st.stop()

    # Load input CSVs
    hist = pd.read_csv(history_file, parse_dates=["date"])  # expects: date, sku, units_sold
    inv  = pd.read_csv(inventory_file, parse_dates=["date"]) # expects: date, sku, on_hand
    if upstream_file:
        upstream = pd.read_csv(upstream_file, parse_dates=["date"]).rename(columns={"units_sold":"yhat"})

    # SKU selection
    skus = hist['sku'].unique().tolist()
    sku = st.selectbox("Select SKU", skus)

    # Filter data for selected SKU
    hist_sku = hist.query("sku == @sku")[['date','units_sold']].rename(columns={'date':'ds','units_sold':'y'})
    inv_sku  = inv.query("sku == @sku")[['date','on_hand']]

    # Prepare event uplift list from text input
events = []
for line in events_text.splitlines():
    parts = line.split(",")
    if len(parts) != 2:
        continue
    date_str, uplift_str = parts[0].strip(), parts[1].strip()
    try:
        ds = pd.to_datetime(date_str)
        uplift = float(uplift_str) / 100.0
        events.append({'ds': ds, 'uplift': uplift})
    except Exception:
        continue

periods = 12  # forecast horizon (weeks)  # forecast horizon (weeks)

    # Forecast logic
    if model_choice == "Prophet":
        df_prophet = hist_sku.copy()
        m = Prophet(weekly_seasonality=True)
        if events:
            event_df = pd.DataFrame(events)
            m.add_regressor('uplift')
            df_prophet = df_prophet.merge(event_df, on='ds', how='left').fillna({'uplift':0})
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=periods, freq='W')
        if events:
            future = future.merge(event_df, on='ds', how='left').fillna({'uplift':0})
        forecast = m.predict(future)[['ds','yhat']].tail(periods)

    elif model_choice == "ARIMA":
        hist_weekly = hist_sku.set_index('ds').resample('W-SUN').sum().reset_index()
        arima_model = ARIMA(hist_weekly['y'], order=(1,1,1))
        arima_res = arima_model.fit()
        fcst_res = arima_res.get_forecast(steps=periods)
        last_date = hist_weekly['ds'].max()
        fcst_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq='W-SUN')
        forecast = pd.DataFrame({'ds': fcst_index, 'yhat': fcst_res.predicted_mean.values})

    else:  # XGBoost
        df = hist_sku.copy()
        df['week'] = df['ds'].dt.isocalendar().week
        df['year'] = df['ds'].dt.year
        for lag in range(1,5):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        df = df.dropna().reset_index(drop=True)
        cutoff_date = df['ds'].max() - pd.Timedelta(weeks=periods)
        train = df[df['ds'] <= cutoff_date]
        test_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(weeks=1), periods=periods, freq='W')
        features = [c for c in train.columns if c.startswith('lag_')] + ['week','year']
        xgb_model = xgb.XGBRegressor(n_estimators=100)
        xgb_model.fit(train[features], train['y'])
        preds = []
        recent_vals = train['y'].tolist()[-4:]
        for dt in test_dates:
            wk, yr = dt.isocalendar().week, dt.year
            feat_vals = recent_vals[-4:] + [wk, yr]
            pred = xgb_model.predict(np.array(feat_vals).reshape(1, -1))[0]
            preds.append({'ds': dt, 'yhat': pred})
            recent_vals.append(pred)
        forecast = pd.DataFrame(preds)

    # Merge inventory
    inv_weekly = (
        inv_sku.set_index('date')
               .resample('W-SUN')
               .last()
               .ffill()
               .reset_index()
               .rename(columns={'date':'ds'})
    )
    merged = forecast.merge(inv_weekly, on='ds', how='left')
    merged['woc'] = merged['on_hand'] / merged['yhat']
    if upstream_file:
        upstream_sku = upstream.query("sku == @sku")[['date','yhat']].rename(columns={'date':'ds'})
        merged = merged.merge(upstream_sku, on='ds', how='left', suffixes=('','_upstream'))

    # Display results
    st.subheader(f"Weekly Sell-In Forecast for SKU: {sku}")
    st.line_chart(merged.set_index('ds')[['yhat','on_hand']])
    st.dataframe(merged[['ds','yhat','on_hand','woc']].round(2))
