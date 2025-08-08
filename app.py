"""
Amazon Replenishment Forecast Streamlit App (Improved Version)

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
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

# Configuration
CONFIG = {
    'AMAZON_BRANDING': {
        'LOGO_URL': "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        'ORANGE': "#FF9900",
        'BLUE': "#146EB4"
    },
    'DEFAULT_FILES': {
        'SALES': "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv",
        'UPSTREAM': "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
    },
    'FORECAST_DEFAULTS': {
        'WOC_TARGET': 4,
        'PERIODS': 12,
        'INIT_INVENTORY': 26730,
        'MIN_WOC': 1,
        'MAX_WOC': 52,
        'MIN_PERIODS': 1,
        'MAX_PERIODS': 104
    }
}

# Optional forecasting libraries with error handling
LIBRARIES = {
    'PLOTLY': False,
    'PROPHET': False,
    'ARIMA': False,
    'XGB': False
}

try:
    import plotly.express as px
    LIBRARIES['PLOTLY'] = True
except ImportError:
    st.warning("⚠️ Plotly not installed. Charts will use basic Streamlit charts.")

try:
    from prophet import Prophet
    LIBRARIES['PROPHET'] = True
except ImportError:
    try:
        from fbprophet import Prophet
        LIBRARIES['PROPHET'] = True
    except ImportError:
        pass

try:
    from statsmodels.tsa.arima.model import ARIMA
    LIBRARIES['ARIMA'] = True
except ImportError:
    pass

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    LIBRARIES['XGB'] = True
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="Amazon Replenishment Forecast", 
    page_icon=CONFIG['AMAZON_BRANDING']['LOGO_URL'], 
    layout="wide"
)

def validate_inputs(init_inv: int, woc_target: int, periods: int) -> bool:
    """Validate user inputs with clear error messages."""
    errors = []
    
    if init_inv < 0:
        errors.append("Initial inventory cannot be negative")
    if not (CONFIG['FORECAST_DEFAULTS']['MIN_WOC'] <= woc_target <= CONFIG['FORECAST_DEFAULTS']['MAX_WOC']):
        errors.append(f"Weeks of Cover must be between {CONFIG['FORECAST_DEFAULTS']['MIN_WOC']}-{CONFIG['FORECAST_DEFAULTS']['MAX_WOC']}")
    if not (CONFIG['FORECAST_DEFAULTS']['MIN_PERIODS'] <= periods <= CONFIG['FORECAST_DEFAULTS']['MAX_PERIODS']):
        errors.append(f"Forecast horizon must be between {CONFIG['FORECAST_DEFAULTS']['MIN_PERIODS']}-{CONFIG['FORECAST_DEFAULTS']['MAX_PERIODS']} weeks")
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        return False
    return True

@st.cache_data
def load_and_clean_sales_data(file_path: str, is_uploaded: bool = False) -> pd.DataFrame:
    """Load and clean sales data with comprehensive error handling."""
    try:
        if is_uploaded:
            df_raw = pd.read_csv(file_path, skiprows=1)
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df_raw = pd.read_csv(file_path, skiprows=1)
        
        if df_raw.empty:
            raise ValueError("Sales file is empty")
        
        # Improved date parsing
        first_col = df_raw.iloc[:, 0].astype(str)
        
        # Try multiple date formats
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{2}-\d{2}-\d{4})'   # MM-DD-YYYY
        ]
        
        dates = None
        for pattern in date_patterns:
            try:
                extracted_dates = first_col.str.extract(pattern)[0]
                dates = pd.to_datetime(extracted_dates, errors='coerce')
                if dates.notna().sum() > 0:
                    break
            except:
                continue
        
        if dates is None or dates.notna().sum() == 0:
            # Fallback: try parsing the entire first column
            dates = pd.to_datetime(first_col.str.split(' - ').str[0].str.strip(), errors='coerce')
        
        df_raw['Week_Start'] = dates
        df_raw.dropna(subset=['Week_Start'], inplace=True)
        
        if df_raw.empty:
            raise ValueError("No valid dates found in the data")
        
        logger.info(f"Successfully loaded {len(df_raw)} rows of sales data")
        return df_raw
        
    except FileNotFoundError as e:
        st.error(f"❌ Sales file not found: {str(e)}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("❌ Sales file contains no data")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Data validation error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error reading sales file: {str(e)}")
        st.stop()

def detect_column_type(df: pd.DataFrame, projection_type: str) -> Tuple[str, str, str]:
    """Detect the correct column for forecasting based on projection type."""
    cols = df.columns.tolist()
    
    if projection_type == 'Units':
        unit_cols = [c for c in cols if re.search(r'unit', c, re.IGNORECASE)]
        y_col = unit_cols[0] if unit_cols else cols[1]
        return y_col, 'Sell-Out Units', 'Units'
    else:
        sales_cols = [c for c in cols if re.search(r'sales', c, re.IGNORECASE)]
        y_col = sales_cols[0] if sales_cols else cols[1]
        return y_col, 'Sell-Out Sales', 'Sales $'

def clean_numerical_data(series: pd.Series) -> pd.Series:
    """Clean numerical data by removing non-numeric characters."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^0-9.]', '', regex=True), 
        errors='coerce'
    ).fillna(0)

@st.cache_data
def extract_product_info(upstream_path: str, is_uploaded: bool = False) -> Tuple[str, str]:
    """Extract product information from upstream forecast file."""
    sku, product = 'N/A', 'N/A'
    
    if not upstream_path:
        return sku, product
    
    try:
        if is_uploaded:
            df_up_hd = pd.read_csv(upstream_path, nrows=1)
        else:
            if not os.path.exists(upstream_path):
                return sku, product
            df_up_hd = pd.read_csv(upstream_path, nrows=1)
        
        # Look for SKU/ASIN columns
        sku_col = next((c for c in df_up_hd.columns if re.search(r'ASIN|SKU', c, re.IGNORECASE)), None)
        name_col = next((c for c in df_up_hd.columns if re.search(r'Name|Title|Product', c, re.IGNORECASE)), None)
        
        if sku_col and not df_up_hd[sku_col].empty:
            sku = str(df_up_hd[sku_col].iloc[0])
        if name_col and not df_up_hd[name_col].empty:
            product = str(df_up_hd[name_col].iloc[0])
            
    except Exception as e:
        logger.warning(f"Could not extract product info: {str(e)}")
    
    return sku, product

def create_xgboost_forecast(df_hist: pd.DataFrame, periods: int) -> pd.Series:
    """Create XGBoost forecast with proper feature engineering."""
    if not LIBRARIES['XGB'] or len(df_hist) < 4:
        # Fallback to exponential smoothing
        return create_exponential_smoothing_forecast(df_hist, periods)
    
    try:
        # Create features
        df_features = df_hist.copy()
        df_features = df_features.sort_values('Week_Start')
        
        # Add lag features
        for lag in [1, 2, 4]:
            if len(df_features) > lag:
                df_features[f'lag_{lag}'] = df_features['y'].shift(lag)
        
        # Add rolling statistics
        df_features['rolling_mean_4'] = df_features['y'].rolling(window=4, min_periods=1).mean()
        df_features['rolling_std_4'] = df_features['y'].rolling(window=4, min_periods=1).std().fillna(0)
        
        # Add time features
        df_features['week_of_year'] = df_features['Week_Start'].dt.isocalendar().week
        df_features['month'] = df_features['Week_Start'].dt.month
        
        # Remove rows with NaN (due to lags)
        df_features = df_features.dropna()
        
        if len(df_features) < 3:
            return create_exponential_smoothing_forecast(df_hist, periods)
        
        # Prepare training data
        feature_cols = [c for c in df_features.columns if c not in ['Week_Start', 'y']]
        X = df_features[feature_cols].fillna(0)
        y = df_features['y']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Generate forecasts
        forecasts = []
        last_row = df_features.iloc[-1:].copy()
        
        for i in range(periods):
            # Prepare features for prediction
            X_pred = last_row[feature_cols].fillna(0)
            X_pred_scaled = scaler.transform(X_pred)
            
            # Make prediction
            pred = model.predict(X_pred_scaled)[0]
            forecasts.append(max(pred, 0))  # Ensure non-negative
            
            # Update features for next prediction (simple approach)
            if len(forecasts) >= 2:
                last_row[feature_cols] = last_row[feature_cols].shift(1, axis=1)
                last_row['lag_1'] = forecasts[-1]
                if len(forecasts) >= 2:
                    last_row['lag_2'] = forecasts[-2]
        
        return pd.Series(forecasts)
    
    except Exception as e:
        logger.warning(f"XGBoost forecast failed: {str(e)}. Using exponential smoothing fallback.")
        return create_exponential_smoothing_forecast(df_hist, periods)

def create_exponential_smoothing_forecast(df_hist: pd.DataFrame, periods: int) -> pd.Series:
    """Create forecast using exponential smoothing as a reliable fallback."""
    if df_hist.empty:
        return pd.Series([0] * periods)
    
    # Simple exponential smoothing with trend
    values = df_hist['y'].values
    alpha = 0.3
    beta = 0.1
    
    if len(values) == 1:
        return pd.Series([values[0]] * periods)
    
    # Initialize
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0
    
    # Smooth historical data
    for i in range(1, len(values)):
        prev_level = level
        level = alpha * values[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    
    # Generate forecast
    forecasts = []
    for i in range(periods):
        forecast = level + (i + 1) * trend
        forecasts.append(max(forecast, 0))  # Ensure non-negative
    
    return pd.Series(forecasts)

@st.cache_data
def generate_forecast(df_hist: pd.DataFrame, periods: int, model_choice: str, future_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate forecast using the selected model."""
    try:
        if model_choice == 'Prophet' and LIBRARIES['PROPHET']:
            # Prophet forecast
            m = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True if len(df_hist) > 52 else False,
                changepoint_prior_scale=0.05
            )
            prophet_df = df_hist.rename(columns={'Week_Start': 'ds'})
            m.fit(prophet_df)
            
            future = pd.DataFrame({'ds': future_idx})
            forecast = m.predict(future)
            
            df_fc = pd.DataFrame({
                'Week_Start': future_idx,
                'yhat': np.maximum(forecast['yhat'].values, 0)  # Ensure non-negative
            })
            
        elif model_choice == 'ARIMA' and LIBRARIES['ARIMA']:
            # ARIMA forecast
            tmp = df_hist.set_index('Week_Start').resample('W').sum().reset_index()
            
            if len(tmp) < 10:
                # Not enough data for ARIMA, use fallback
                forecasts = create_exponential_smoothing_forecast(df_hist, periods)
            else:
                try:
                    model = ARIMA(tmp['y'], order=(1, 1, 1))
                    fitted_model = model.fit()
                    forecast_result = fitted_model.get_forecast(steps=periods)
                    forecasts = np.maximum(forecast_result.predicted_mean.values, 0)
                except:
                    forecasts = create_exponential_smoothing_forecast(df_hist, periods)
            
            df_fc = pd.DataFrame({
                'Week_Start': future_idx,
                'yhat': forecasts
            })
            
        elif model_choice == 'XGBoost':
            # XGBoost forecast
            forecasts = create_xgboost_forecast(df_hist, periods)
            df_fc = pd.DataFrame({
                'Week_Start': future_idx,
                'yhat': forecasts.values
            })
            
        else:
            # Fallback forecast
            forecasts = create_exponential_smoothing_forecast(df_hist, periods)
            df_fc = pd.DataFrame({
                'Week_Start': future_idx,
                'yhat': forecasts.values
            })
        
        logger.info(f"Generated {model_choice} forecast for {periods} periods")
        return df_fc
        
    except Exception as e:
        logger.error(f"Forecast generation failed with {model_choice}: {str(e)}")
        # Ultimate fallback
        forecasts = create_exponential_smoothing_forecast(df_hist, periods)
        return pd.DataFrame({
            'Week_Start': future_idx,
            'yhat': forecasts.values
        })

def calculate_replenishment(forecast_df: pd.DataFrame, init_inventory: int, woc_target: int, forecast_col: str) -> pd.DataFrame:
    """Calculate replenishment needs based on forecast."""
    on_hand = []
    replenishment = []
    prev_on = init_inventory
    
    for _, row in forecast_df.iterrows():
        demand = max(row[forecast_col], 0)  # Ensure non-negative demand
        target = demand * woc_target
        replen_needed = max(target - prev_on, 0)
        
        on_hand.append(int(prev_on))
        replenishment.append(int(replen_needed))
        
        # Update inventory for next period
        prev_on = prev_on + replen_needed - demand
    
    forecast_df = forecast_df.copy()
    forecast_df['On_Hand_Begin'] = on_hand
    forecast_df['Replenishment'] = replenishment
    
    # Calculate Weeks of Cover with safe division
    forecast_df['Weeks_Of_Cover'] = np.where(
        forecast_df[forecast_col] > 0,
        (forecast_df['On_Hand_Begin'] / forecast_df[forecast_col]).round(2),
        999.99  # Represents "infinite" weeks when demand is zero
    )
    
    return forecast_df

def load_amazon_sellout_forecast(upstream_path: str, is_uploaded: bool = False) -> pd.DataFrame:
    """Load and parse Amazon sellout forecast data."""
    if not upstream_path:
        return pd.DataFrame()
    
    try:
        if is_uploaded:
            df_up = pd.read_csv(upstream_path, skiprows=1)
        else:
            if not os.path.exists(upstream_path):
                return pd.DataFrame()
            df_up = pd.read_csv(upstream_path, skiprows=1)
        
        records = []
        current_year = datetime.now().year
        
        for col in df_up.columns:
            if col.startswith('Week '):
                # Extract date from column name
                date_match = re.search(r"\((\d{1,2} [A-Za-z]+)", col)
                if date_match:
                    try:
                        date_str = f"{date_match.group(1)} {current_year}"
                        parsed_date = pd.to_datetime(date_str, format='%d %b %Y', errors='coerce')
                        
                        if parsed_date is not None:
                            # Clean and convert value
                            raw_value = str(df_up[col].iloc[0]) if not df_up[col].empty else "0"
                            clean_value = re.sub(r'[^\d.]', '', raw_value.replace(',', ''))
                            numeric_value = pd.to_numeric(clean_value, errors='coerce')
                            
                            if pd.notna(numeric_value):
                                records.append({
                                    'Week_Start': parsed_date,
                                    'Amazon_Sellout_Forecast': int(round(numeric_value))
                                })
                    except Exception as e:
                        logger.warning(f"Could not parse date from column {col}: {str(e)}")
        
        return pd.DataFrame(records) if records else pd.DataFrame()
        
    except Exception as e:
        logger.warning(f"Could not load Amazon sellout forecast: {str(e)}")
        return pd.DataFrame()

def create_visualization(df_fc: pd.DataFrame, periods: int, projection_type: str, y_label: str):
    """Create forecast visualization."""
    metrics = ['Replenishment']
    if 'Amazon_Sellout_Forecast' in df_fc.columns:
        metrics.insert(0, 'Amazon_Sellout_Forecast')
    
    st.subheader(f"📊 {periods}-Week Replenishment Plan ({projection_type})")
    
    if LIBRARIES['PLOTLY']:
        try:
            fig = px.line(
                df_fc, 
                x='Week_Start', 
                y=metrics,
                labels={'value': y_label, 'variable': 'Metric'},
                title='Replenishment vs Demand Forecast'
            )
            fig.update_xaxes(tickformat='%d-%m-%Y')
            fig.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.warning(f"Plotly chart failed: {str(e)}. Using Streamlit chart.")
            st.line_chart(df_fc.set_index('Week_Start')[metrics])
    else:
        st.line_chart(df_fc.set_index('Week_Start')[metrics])

def main():
    """Main application function."""
    # Header with branding
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; margin-bottom: 30px;'>
            <img src='{CONFIG['AMAZON_BRANDING']['LOGO_URL']}' width='100' style='margin-right: 20px;'>
            <h1 style='color:{CONFIG['AMAZON_BRANDING']['BLUE']}; margin: 0;'>
                Amazon Replenishment Forecast
            </h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    st.sidebar.header("📊 Data Inputs & Settings")
    
    # File uploads
    sales_file = st.sidebar.file_uploader(
        "📈 Sales History CSV", 
        type=["csv"],
        help="Upload your historical sales data CSV file"
    )
    
    fcst_file = st.sidebar.file_uploader(
        "🔮 Amazon Sell-Out Forecast CSV", 
        type=["csv"],
        help="Optional: Upload Amazon's sellout forecast for comparison"
    )
    
    # Configuration inputs
    projection_type = st.sidebar.selectbox(
        "📊 Projection Type", 
        ["Units", "Sales $"],
        help="Choose whether to forecast in units or sales dollars"
    )
    
    init_inv = st.sidebar.number_input(
        "📦 Current On-Hand Inventory (units)", 
        min_value=0, 
        value=CONFIG['FORECAST_DEFAULTS']['INIT_INVENTORY'], 
        step=1,
        help="Current inventory level in units"
    )
    
    # Model selection
    available_models = []
    if LIBRARIES['PROPHET']:
        available_models.append("Prophet")
    if LIBRARIES['ARIMA']:
        available_models.append("ARIMA")
    if LIBRARIES['XGB']:
        available_models.append("XGBoost")
    
    # Always include fallback
    available_models.append("Exponential Smoothing")
    
    if not available_models:
        st.error("❌ No forecasting libraries available. Please install prophet, statsmodels, or xgboost.")
        st.stop()
    
    model_choice = st.sidebar.selectbox(
        "🤖 Forecast Model", 
        available_models,
        help="Choose the forecasting algorithm"
    )
    
    # Forecast parameters
    woc_target = st.sidebar.slider(
        "📅 Target Weeks of Cover", 
        CONFIG['FORECAST_DEFAULTS']['MIN_WOC'], 
        CONFIG['FORECAST_DEFAULTS']['MAX_WOC'], 
        CONFIG['FORECAST_DEFAULTS']['WOC_TARGET'],
        help="Target inventory level in weeks of demand coverage"
    )
    
    periods = st.sidebar.number_input(
        "🔭 Forecast Horizon (weeks)", 
        min_value=CONFIG['FORECAST_DEFAULTS']['MIN_PERIODS'], 
        max_value=CONFIG['FORECAST_DEFAULTS']['MAX_PERIODS'], 
        value=CONFIG['FORECAST_DEFAULTS']['PERIODS'],
        help="Number of weeks to forecast ahead"
    )
    
    # Validation
    if not validate_inputs(init_inv, woc_target, periods):
        st.stop()
    
    st.sidebar.markdown("---")
    
    # Run forecast button
    if not st.sidebar.button("🚀 Run Forecast", type="primary"):
        st.info("👆 Configure your settings and click **'Run Forecast'** to generate your replenishment plan.")
        
        # Show library status
        with st.expander("📚 Available Libraries"):
            for lib, available in LIBRARIES.items():
                status = "✅ Available" if available else "❌ Not installed"
                st.write(f"**{lib}**: {status}")
        
        st.stop()
    
    # Determine file paths
    sales_path = sales_file if sales_file else CONFIG['DEFAULT_FILES']['SALES']
    upstream_path = fcst_file if fcst_file else CONFIG['DEFAULT_FILES']['UPSTREAM']
    
    if not sales_path:
        st.error("❌ Sales history file is required.")
        st.stop()
    
    # Load and process data
    with st.spinner('🔄 Loading and processing sales data...'):
        df_raw = load_and_clean_sales_data(sales_path, is_uploaded=bool(sales_file))
        
        # Extract product information
        sku, product = extract_product_info(upstream_path, is_uploaded=bool(fcst_file))
        
        # Display product info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📦 Product", product)
        with col2:
            st.metric("🏷️ SKU/ASIN", sku)
    
    # Determine data column
    y_col, forecast_label, y_label = detect_column_type(df_raw, projection_type)
    
    # Clean historical data
    with st.spinner('🧹 Cleaning historical data...'):
        df_hist = pd.DataFrame({
            'Week_Start': df_raw['Week_Start'],
            'y': clean_numerical_data(df_raw[y_col])
        })
        
        # Filter to historical data only
        df_hist = df_hist[df_hist['Week_Start'] <= pd.to_datetime(datetime.now().date())]
        
        if df_hist.empty:
            st.error("❌ No valid historical data found.")
            st.stop()
        
        # Remove duplicate dates and sort
        df_hist = df_hist.drop_duplicates(subset=['Week_Start']).sort_values('Week_Start')
    
    # Generate forecast
    with st.spinner(f'🔮 Generating {model_choice} forecast...'):
        # Create future date index
        last_hist = df_hist['Week_Start'].max()
        start_fc = max(
            last_hist + timedelta(weeks=1),
            pd.to_datetime(datetime.now().date()) + timedelta(weeks=1)
        )
        future_idx = pd.date_range(start=start_fc, periods=periods, freq='W')
        
        # Generate forecast
        df_fc = generate_forecast(df_hist, periods, model_choice, future_idx)
        df_fc[forecast_label] = df_fc['yhat'].round(0).astype(int)
    
    # Calculate replenishment
    with st.spinner('📊 Calculating replenishment needs...'):
        df_fc = calculate_replenishment(df_fc, init_inv, woc_target, forecast_label)
    
    # Load Amazon sellout forecast if available
    if upstream_path:
        with st.spinner('🔄 Loading Amazon sellout forecast...'):
            amazon_forecast = load_amazon_sellout_forecast(upstream_path, is_uploaded=bool(fcst_file))
            if not amazon_forecast.empty:
                df_fc = df_fc.merge(amazon_forecast, on='Week_Start', how='left')
    
    # Format dates for display
    df_fc['Date'] = df_fc['Week_Start'].dt.strftime('%d-%m-%Y')
    
    # Create visualization
    create_visualization(df_fc, periods, projection_type, y_label)
    
    # Display results table
    st.subheader("📋 Detailed Replenishment Plan")
    
    display_columns = ['Date', forecast_label, 'On_Hand_Begin', 'Replenishment', 'Weeks_Of_Cover']
    if 'Amazon_Sellout_Forecast' in df_fc.columns:
        display_columns.insert(1, 'Amazon_Sellout_Forecast')
    
    # Format the dataframe for better display
    df_display = df_fc[display_columns].copy()
    
    # Add summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_replenishment = df_fc['Replenishment'].sum()
        st.metric("📦 Total Replenishment", f"{total_replenishment:,} units")
    
    with col2:
        avg_weekly_demand = df_fc[forecast_label].mean()
        st.metric("📊 Avg Weekly Demand", f"{avg_weekly_demand:,.0f} units")
    
    with col3:
        min_weeks_cover = df_fc['Weeks_Of_Cover'].min()
        st.metric("⚠️ Min Weeks Cover", f"{min_weeks_cover:.1f} weeks")
    
    with col4:
        max_inventory = df_fc['On_Hand_Begin'].max()
        st.metric("📈 Peak Inventory", f"{max_inventory:,} units")
    
    # Display the table
    st.dataframe(df_display, use_container_width=True)
    
    # Download button for results
    csv = df_fc.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"replenishment_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align:center; color:gray; margin-top:20px;'>
            <p>📊 Generated with {model_choice} • 
            ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} • 
            &copy; {datetime.now().year} Amazon Internal Tool</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()"""
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

# Optional forecasting libraries
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
init_inv = st.sidebar.number_input(
    "Current On-Hand Inventory (units)", min_value=0, value=26730, step=1
)

# Forecast model selection
model_opts = []
if PROPHET_INSTALLED:
    model_opts.append("Prophet")
if ARIMA_INSTALLED:
    model_opts.append("ARIMA")
if XGB_INSTALLED:
    model_opts.append("XGBoost")
if not model_opts:
    st.error("Install at least one forecasting engine: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Forecast Model", model_opts)

# Parameters: WOC and forecast horizon
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = st.sidebar.number_input(
    "Forecast Horizon (weeks)", min_value=4, max_value=52, value=12
)

st.markdown("---")
# Run forecast trigger
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to run the replenishment plan.")
    st.stop()

# Resolve file paths
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_upstream = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
sales_path = sales_file if sales_file else (default_sales if os.path.exists(default_sales) else None)
upstream_path = fcst_file if fcst_file else (default_upstream if os.path.exists(default_upstream) else None)
if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

# Load sales history
df_raw = pd.read_csv(sales_path, skiprows=1)
# Parse week start date
df_raw['Week_Start'] = pd.to_datetime(
    df_raw.iloc[:,0].astype(str).str.split(' - ').str[0].str.strip(), errors='coerce'
)
df_raw.dropna(subset=['Week_Start'], inplace=True)

# Auto-detect product info from sell-out forecast file
sku, product = 'N/A', 'N/A'
if upstream_path:
    try:
        df_up_hd = pd.read_csv(upstream_path, nrows=1)
        sku_col = next((c for c in df_up_hd.columns if re.search(r'ASIN|SKU', c, re.IGNORECASE)), None)
        name_col = next((c for c in df_up_hd.columns if re.search(r'Name|Title|Product', c, re.IGNORECASE)), None)
        if sku_col:
            sku = df_up_hd[sku_col].iloc[0]
        if name_col:
            product = df_up_hd[name_col].iloc[0]
    except Exception:
        pass
# Display product details
st.markdown(
    f"**Product:** {product}  <br>**SKU:** {sku}", unsafe_allow_html=True
)

# Determine metric column for historical data
cols = df_raw.columns.tolist()
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

# Clean historical series
df_hist = pd.DataFrame({
    'Week_Start': df_raw['Week_Start'],
    'y': pd.to_numeric(
        df_raw[y_col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce'
    ).fillna(0)
})
df_hist = df_hist[df_hist['Week_Start'] <= pd.to_datetime(datetime.now().date())]
if df_hist.empty:
    st.error("No valid historical data.")
    st.stop()

# Forecast date range
last_hist = df_hist['Week_Start'].max()
start_fc = max(
    last_hist + timedelta(weeks=1),
    pd.to_datetime(datetime.now().date()) + timedelta(weeks=1)
)
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

# Round and rename forecast column
df_fc[forecast_label] = df_fc['yhat'].round(0).astype(int)

# Replenishment logic based on manual init_inv
on_hand = []
replen = []
prev_on = init_inv
for _, row in df_fc.iterrows():
    D = row[forecast_label]
    target = D * woc_target
    Q = max(target - prev_on, 0)
    on_hand.append(int(prev_on))
    replen.append(int(Q))
    prev_on = prev_on + Q - D

df_fc['On_Hand_Begin'] = on_hand
df_fc['Replenishment'] = replen
# Dynamic Weeks of Cover based on on-hand and demand
df_fc['Weeks_Of_Cover'] = (
    df_fc['On_Hand_Begin'] / df_fc[forecast_label]
).replace([np.inf, -np.inf], np.nan).round(2)

# Merge Amazon sell-out forecast if available
if upstream_path:
    df_up = pd.read_csv(upstream_path, skiprows=1)
    rec = []
    for c in df_up.columns:
        if c.startswith('Week '):
            m = re.search(r"\((\d{1,2} [A-Za-z]+)", c)
            if m:
                dt = pd.to_datetime(
                    m.group(1) + ' ' + str(datetime.now().year),
                    format='%d %b %Y', errors='coerce'
                )
                val = pd.to_numeric(
                    str(df_up[c].iloc[0]).replace(',', ''), errors='coerce'
                )
                rec.append({'Week_Start': dt, 'Amazon_Sellout_Forecast': int(round(val))})
    if rec:
        df_fc = df_fc.merge(pd.DataFrame(rec), on='Week_Start', how='left')

# Format date for display
df_fc['Date'] = df_fc['Week_Start'].dt.strftime('%d-%m-%Y')

# Plot results
st.subheader(f"{periods}-Week Replenishment Plan ({projection_type})")
metrics = ['Replenishment']
if 'Amazon_Sellout_Forecast' in df_fc.columns:
    metrics.insert(0, 'Amazon_Sellout_Forecast')

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

# Display table and footer
cols = ['Date'] + metrics + ['On_Hand_Begin', 'Weeks_Of_Cover']
st.dataframe(df_fc[cols])

st.markdown(
    f"<div style='text-align:center; color:gray; margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
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

# Optional forecasting libraries
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
init_inv = st.sidebar.number_input(
    "Current On-Hand Inventory (units)", min_value=0, value=26730, step=1
)

# Forecast model selection
model_opts = []
if PROPHET_INSTALLED:
    model_opts.append("Prophet")
if ARIMA_INSTALLED:
    model_opts.append("ARIMA")
if XGB_INSTALLED:
    model_opts.append("XGBoost")
if not model_opts:
    st.error("Install at least one forecasting engine: prophet, statsmodels, or xgboost.")
    st.stop()
model_choice = st.sidebar.selectbox("Forecast Model", model_opts)

# Parameters: WOC and forecast horizon
woc_target = st.sidebar.slider("Target Weeks of Cover", 1, 12, 4)
periods = st.sidebar.number_input(
    "Forecast Horizon (weeks)", min_value=4, max_value=52, value=12
)

st.markdown("---")
# Run forecast trigger
if not st.button("Run Forecast"):
    st.info("Click 'Run Forecast' to run the replenishment plan.")
    st.stop()

# Resolve file paths
default_sales = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
default_upstream = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
sales_path = sales_file if sales_file else (default_sales if os.path.exists(default_sales) else None)
upstream_path = fcst_file if fcst_file else (default_upstream if os.path.exists(default_upstream) else None)
if not sales_path:
    st.error("Sales history file is required.")
    st.stop()

# Load sales history
df_raw = pd.read_csv(sales_path, skiprows=1)
# Parse week start date
df_raw['Week_Start'] = pd.to_datetime(
    df_raw.iloc[:,0].astype(str).str.split(' - ').str[0].str.strip(), errors='coerce'
)
df_raw.dropna(subset=['Week_Start'], inplace=True)

# Auto-detect product info from sell-out forecast file
sku, product = 'N/A', 'N/A'
if upstream_path:
    try:
        df_up_hd = pd.read_csv(upstream_path, nrows=1)
        sku_col = next((c for c in df_up_hd.columns if re.search(r'ASIN|SKU', c, re.IGNORECASE)), None)
        name_col = next((c for c in df_up_hd.columns if re.search(r'Name|Title|Product', c, re.IGNORECASE)), None)
        if sku_col:
            sku = df_up_hd[sku_col].iloc[0]
        if name_col:
            product = df_up_hd[name_col].iloc[0]
    except Exception:
        pass
# Display product details
st.markdown(
    f"**Product:** {product}  <br>**SKU:** {sku}", unsafe_allow_html=True
)

# Determine metric column for historical data
cols = df_raw.columns.tolist()
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

# Clean historical series
df_hist = pd.DataFrame({
    'Week_Start': df_raw['Week_Start'],
    'y': pd.to_numeric(
        df_raw[y_col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce'
    ).fillna(0)
})
df_hist = df_hist[df_hist['Week_Start'] <= pd.to_datetime(datetime.now().date())]
if df_hist.empty:
    st.error("No valid historical data.")
    st.stop()

# Forecast date range
last_hist = df_hist['Week_Start'].max()
start_fc = max(
    last_hist + timedelta(weeks=1),
    pd.to_datetime(datetime.now().date()) + timedelta(weeks=1)
)
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

# Round and rename forecast column
df_fc[forecast_label] = df_fc['yhat'].round(0).astype(int)

# Replenishment logic based on manual init_inv
on_hand = []
replen = []
prev_on = init_inv
for _, row in df_fc.iterrows():
    D = row[forecast_label]
    target = D * woc_target
    Q = max(target - prev_on, 0)
    on_hand.append(int(prev_on))
    replen.append(int(Q))
    prev_on = prev_on + Q - D

df_fc['On_Hand_Begin'] = on_hand
df_fc['Replenishment'] = replen
# Dynamic Weeks of Cover based on on-hand and demand
df_fc['Weeks_Of_Cover'] = (
    df_fc['On_Hand_Begin'] / df_fc[forecast_label]
).replace([np.inf, -np.inf], np.nan).round(2)

# Merge Amazon sell-out forecast if available
if upstream_path:
    df_up = pd.read_csv(upstream_path, skiprows=1)
    rec = []
    for c in df_up.columns:
        if c.startswith('Week '):
            m = re.search(r"\((\d{1,2} [A-Za-z]+)", c)
            if m:
                dt = pd.to_datetime(
                    m.group(1) + ' ' + str(datetime.now().year),
                    format='%d %b %Y', errors='coerce'
                )
                val = pd.to_numeric(
                    str(df_up[c].iloc[0]).replace(',', ''), errors='coerce'
                )
                rec.append({'Week_Start': dt, 'Amazon_Sellout_Forecast': int(round(val))})
    if rec:
        df_fc = df_fc.merge(pd.DataFrame(rec), on='Week_Start', how='left')

# Format date for display
df_fc['Date'] = df_fc['Week_Start'].dt.strftime('%d-%m-%Y')

# Plot results
st.subheader(f"{periods}-Week Replenishment Plan ({projection_type})")
metrics = ['Replenishment']
if 'Amazon_Sellout_Forecast' in df_fc.columns:
    metrics.insert(0, 'Amazon_Sellout_Forecast')

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

# Display table and footer
cols = ['Date'] + metrics + ['On_Hand_Begin', 'Weeks_Of_Cover']
st.dataframe(df_fc[cols])

st.markdown(
    f"<div style='text-align:center; color:gray; margin-top:20px;'>&copy; {datetime.now().year} Amazon Internal Tool</div>",
    unsafe_allow_html=True
)
