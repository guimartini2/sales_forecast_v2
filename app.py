import streamlit as st
import pandas as pd
import numpy as np
# Prophet import with fallback
try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        raise ImportError(
            "Prophet library not found. Install with: pip install prophet (or pip install fbprophet)"
        )
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

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
model_choice = st.sidebar.selectbox("Select forecast model", ["Prophet", "ARIMA", "XGBoost"])
woc_target = st.sidebar.slider("Target Weeks of Cover", 1.0, 12.0, 4.0, step=0.5)
events_df = st.sidebar.experimental_data_editor(
    pd.DataFrame(columns=["date", "uplift_pct"]), key="events"
)

if st.sidebar.button("Run Forecast"):
    # Validate required uploads
    if not history_file or not inventory_file:
        st.error("Please upload both sell-out history and inventory files to proceed.")
    else:
        # Load input CSVs
        hist = pd.read_csv(history_file, parse_dates=["date"])  # expects columns: date, sku, units_sold
        inv  = pd.read_csv(inventory_file, parse_dates=["date"])  # expects columns: date, sku, on_hand
        if upstream_file:
            upstream = pd.read_csv(upstream_file, parse_dates=["date"]).rename(columns={"units_sold":"yhat"})

        # SKU selection
        skus = hist['sku'].unique().tolist()
        sku = st.selectbox("Select SKU", skus)

        # Filter data for selected SKU
        hist_sku = hist.query("sku == @sku")[['date', 'units_sold']].rename(columns={'date':'ds', 'units_sold':'y'})
        inv_sku  = inv.query("sku == @sku")[['date','on_hand']]

        # Prepare event regressor list for Prophet
        events = []
        for _, row in events_df.iterrows():
            try:
                ds = pd.to_datetime(row['date'])
                uplift = float(row['uplift_pct']) / 100.0
                events.append({'ds': ds, 'uplift': uplift})
            except Exception:
                continue

        # Forecast horizon: next 12 weeks
        periods = 12

        # Run chosen model
        if model_choice == "Prophet":
            df_prophet = hist_sku.copy()
            m = Prophet(weekly_seasonality=True)
            # add uplift regressor if events provided
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
            # Resample weekly
            hist_weekly = hist_sku.set_index('ds').resample('W-SUN').sum().reset_index()
            arima_model = ARIMA(hist_weekly['y'], order=(1,1,1))
            arima_res = arima_model.fit()
            fcst_res = arima_res.get_forecast(steps=periods)
            # build forecast DataFrame
            last_date = hist_weekly['ds'].max()
            fcst_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq='W-SUN')
            forecast = pd.DataFrame({'ds': fcst_index, 'yhat': fcst_res.predicted_mean.values})

        else:  # XGBoost
            df = hist_sku.copy()
            df['week'] = df['ds'].dt.isocalendar().week
            df['year'] = df['ds'].dt.year
            # create lag features
            for lag in range(1,5):
                df[f'lag_{lag}'] = df['y'].shift(lag)
            df = df.dropna().reset_index(drop=True)
            # training set cutoff
            cutoff_date = df['ds'].max() - pd.Timedelta(weeks=periods)
            train = df[df['ds'] <= cutoff_date]
            # define test dates
            test_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(weeks=1), periods=periods, freq='W')
            # train XGBoost
            features = [c for c in train.columns if c.startswith('lag_')] + ['week','year']
            xgb_model = xgb.XGBRegressor(n_estimators=100)
            xgb_model.fit(train[features], train['y'])
            # iterative forecast
            preds = []
            recent_vals = train['y'].tolist()[-4:]
            for dt in test_dates:
                wk = dt.isocalendar().week
                yr = dt.year
                feat_vals = recent_vals[-4:] + [wk, yr]
                pred = xgb_model.predict(np.array(feat_vals).reshape(1, -1))[0]
                preds.append({'ds': dt, 'yhat': pred})
                recent_vals.append(pred)
            forecast = pd.DataFrame(preds)

        # Merge with inventory and optional upstream forecast
        # Prepare weekly inventory on-hand
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
        # if upstream forecast provided, merge and display
        if upstream_file:
            upstream_sku = upstream.query("sku == @sku")[['date','yhat']].rename(columns={'date':'ds'})
            merged = merged.merge(upstream_sku, on='ds', how='left', suffixes=('','_upstream'))

        # Display results
        st.subheader(f"Weekly Sell-In Forecast for SKU: {sku}")
        chart_df = merged.set_index('ds')[['yhat','on_hand']]
        st.line_chart(chart_df)
        st.dataframe(merged[['ds','yhat','on_hand','woc']].round(2))
