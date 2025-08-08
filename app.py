"""
Improved Amazon Replenishment Forecast Streamlit App

Key improvements:
- Better error handling and validation
- More modular architecture 
- Enhanced data processing
- Improved UI/UX
- Better performance with caching
- Comprehensive logging
- More robust forecasting models
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np

# Configuration using dataclass for better structure
@dataclass
class Config:
    # Amazon Branding
    LOGO_URL: str = "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
    ORANGE: str = "#FF9900"
    BLUE: str = "#146EB4"
    LIGHT_BLUE: str = "#E7F3FF"
    
    # Default file paths
    DEFAULT_SALES_PATH: str = "/mnt/data/Sales_Week_Manufacturing_Retail_UnitedStates_Custom_1-1-2024_12-31-2024.csv"
    DEFAULT_UPSTREAM_PATH: str = "/mnt/data/Forecasting_ASIN_Retail_MeanForecast_UnitedStates.csv"
    
    # Forecast parameters
    WOC_TARGET: int = 4
    PERIODS: int = 12
    INIT_INVENTORY: int = 26730
    MIN_WOC: int = 1
    MAX_WOC: int = 52
    MIN_PERIODS: int = 1
    MAX_PERIODS: int = 104
    
    # Data validation
    MAX_FILE_SIZE_MB: int = 100
    MIN_HISTORICAL_WEEKS: int = 4

CONFIG = Config()

# Library availability checker
class LibraryManager:
    def __init__(self):
        self.libraries = {}
        self._check_libraries()
    
    def _check_libraries(self):
        """Check which optional libraries are available."""
        # Plotly
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            self.libraries['plotly'] = {'available': True, 'modules': {'px': px, 'go': go}}
        except ImportError:
            self.libraries['plotly'] = {'available': False, 'modules': {}}
        
        # Prophet
        try:
            from prophet import Prophet
            self.libraries['prophet'] = {'available': True, 'modules': {'Prophet': Prophet}}
        except ImportError:
            try:
                from fbprophet import Prophet
                self.libraries['prophet'] = {'available': True, 'modules': {'Prophet': Prophet}}
            except ImportError:
                self.libraries['prophet'] = {'available': False, 'modules': {}}
        
        # ARIMA
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.libraries['statsmodels'] = {
                'available': True, 
                'modules': {'ARIMA': ARIMA, 'ExponentialSmoothing': ExponentialSmoothing}
            }
        except ImportError:
            self.libraries['statsmodels'] = {'available': False, 'modules': {}}
        
        # XGBoost and sklearn
        try:
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            self.libraries['ml'] = {
                'available': True, 
                'modules': {
                    'xgb': xgb, 
                    'StandardScaler': StandardScaler,
                    'mean_absolute_error': mean_absolute_error,
                    'mean_squared_error': mean_squared_error
                }
            }
        except ImportError:
            self.libraries['ml'] = {'available': False, 'modules': {}}
    
    def is_available(self, library: str) -> bool:
        return self.libraries.get(library, {}).get('available', False)
    
    def get_module(self, library: str, module: str):
        return self.libraries.get(library, {}).get('modules', {}).get(module)
    
    def get_available_models(self) -> List[str]:
        """Get list of available forecasting models."""
        models = ['Linear Trend', 'Exponential Smoothing']
        
        if self.is_available('prophet'):
            models.append('Prophet')
        if self.is_available('statsmodels'):
            models.append('ARIMA')
        if self.is_available('ml'):
            models.append('XGBoost')
            
        return models

# Initialize library manager
LIB_MANAGER = LibraryManager()

# Enhanced logging setup
def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Streamlit configuration
st.set_page_config(
    page_title="Amazon Replenishment Forecast", 
    page_icon="📦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_custom_css():
    """Load custom CSS for better UI."""
    st.markdown(f"""
    <style>
        .main-header {{
            background: linear-gradient(90deg, {CONFIG.BLUE} 0%, {CONFIG.ORANGE} 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }}
        
        .metric-card {{
            background: {CONFIG.LIGHT_BLUE};
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {CONFIG.BLUE};
        }}
        
        .warning-box {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        .error-box {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        .success-box {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)

class DataValidator:
    """Enhanced data validation class."""
    
    @staticmethod
    def validate_file_size(file) -> bool:
        """Check if uploaded file is within size limits."""
        if file is None:
            return True
        
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > CONFIG.MAX_FILE_SIZE_MB:
            st.error(f"❌ File size ({file_size_mb:.1f}MB) exceeds limit of {CONFIG.MAX_FILE_SIZE_MB}MB")
            return False
        return True
    
    @staticmethod
    def validate_forecast_inputs(init_inv: int, woc_target: int, periods: int) -> Tuple[bool, List[str]]:
        """Validate user inputs with detailed error messages."""
        errors = []
        
        if init_inv < 0:
            errors.append("Initial inventory cannot be negative")
        elif init_inv > 10_000_000:
            errors.append("Initial inventory seems unreasonably high (>10M units)")
            
        if not (CONFIG.MIN_WOC <= woc_target <= CONFIG.MAX_WOC):
            errors.append(f"Weeks of Cover must be between {CONFIG.MIN_WOC}-{CONFIG.MAX_WOC}")
            
        if not (CONFIG.MIN_PERIODS <= periods <= CONFIG.MAX_PERIODS):
            errors.append(f"Forecast horizon must be between {CONFIG.MIN_PERIODS}-{CONFIG.MAX_PERIODS} weeks")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, file_type: str) -> Tuple[bool, List[str]]:
        """Validate loaded dataframe."""
        errors = []
        
        if df.empty:
            errors.append(f"{file_type} file contains no data")
            return False, errors
        
        if len(df.columns) < 2:
            errors.append(f"{file_type} file must have at least 2 columns")
            
        # Check for reasonable data size
        if len(df) > 10000:
            errors.append(f"{file_type} file has too many rows ({len(df)}). Maximum 10,000 rows supported.")
        
        return len(errors) == 0, errors

class DataProcessor:
    """Enhanced data processing with better error handling."""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_and_clean_sales_data(file_path: str, is_uploaded: bool = False) -> pd.DataFrame:
        """Load and clean sales data with comprehensive error handling."""
        try:
            # Load data
            if is_uploaded:
                df_raw = pd.read_csv(file_path, skiprows=1)
            else:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                df_raw = pd.read_csv(file_path, skiprows=1)
            
            # Validate basic structure
            is_valid, errors = DataValidator.validate_dataframe(df_raw, "Sales")
            if not is_valid:
                for error in errors:
                    st.error(f"❌ {error}")
                st.stop()
            
            # Enhanced date parsing
            df_raw = DataProcessor._parse_dates(df_raw)
            
            # Remove rows with invalid dates
            df_raw = df_raw.dropna(subset=['Week_Start'])
            
            if df_raw.empty:
                raise ValueError("No valid dates found in the data")
            
            logger.info(f"Successfully loaded {len(df_raw)} rows of sales data")
            return df_raw
            
        except Exception as e:
            logger.error(f"Error loading sales data: {str(e)}")
            st.error(f"❌ Error loading sales file: {str(e)}")
            st.stop()
    
    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced date parsing with multiple format support."""
        first_col = df.iloc[:, 0].astype(str)
        
        # Multiple date parsing strategies
        date_strategies = [
            # Strategy 1: Direct datetime parsing
            lambda x: pd.to_datetime(x, errors='coerce'),
            
            # Strategy 2: Extract date patterns
            lambda x: DataProcessor._extract_date_patterns(x),
            
            # Strategy 3: Split and parse
            lambda x: pd.to_datetime(x.str.split(' - ').str[0].str.strip(), errors='coerce')
        ]
        
        dates = None
        for i, strategy in enumerate(date_strategies):
            try:
                dates = strategy(first_col)
                valid_dates = dates.notna().sum()
                if valid_dates > 0:
                    logger.info(f"Date parsing strategy {i+1} successful: {valid_dates} valid dates")
                    break
            except Exception as e:
                logger.warning(f"Date parsing strategy {i+1} failed: {str(e)}")
                continue
        
        df['Week_Start'] = dates
        return df
    
    @staticmethod
    def _extract_date_patterns(series: pd.Series) -> pd.Series:
        """Extract dates using regex patterns."""
        date_patterns = [
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),  # YYYY-MM-DD
            (r'(\d{2}/\d{2}/\d{4})', '%m/%d/%Y'),  # MM/DD/YYYY
            (r'(\d{2}-\d{2}-\d{4})', '%m-%d-%Y'),  # MM-DD-YYYY
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),  # M/D/YYYY
        ]
        
        for pattern, date_format in date_patterns:
            try:
                extracted = series.str.extract(pattern)[0]
                dates = pd.to_datetime(extracted, format=date_format, errors='coerce')
                if dates.notna().sum() > 0:
                    return dates
            except:
                continue
                
        return pd.Series([pd.NaT] * len(series))
    
    @staticmethod
    def detect_column_type(df: pd.DataFrame, projection_type: str) -> Tuple[str, str, str]:
        """Enhanced column detection with better pattern matching."""
        cols = df.columns.tolist()
        
        if projection_type == 'Units':
            # Look for unit columns
            unit_patterns = [r'unit', r'qty', r'quantity', r'volume', r'pieces']
            for pattern in unit_patterns:
                unit_cols = [c for c in cols if re.search(pattern, c, re.IGNORECASE)]
                if unit_cols:
                    return unit_cols[0], 'Sell-Out Units', 'Units'
            
            # Fallback to second column
            return cols[1] if len(cols) > 1 else cols[0], 'Sell-Out Units', 'Units'
        else:
            # Look for sales/revenue columns
            sales_patterns = [r'sales', r'revenue', r'amount', r'value', r'\$']
            for pattern in sales_patterns:
                sales_cols = [c for c in cols if re.search(pattern, c, re.IGNORECASE)]
                if sales_cols:
                    return sales_cols[0], 'Sell-Out Sales', 'Sales $'
            
            # Fallback to second column
            return cols[1] if len(cols) > 1 else cols[0], 'Sell-Out Sales', 'Sales $'
    
    @staticmethod
    def clean_numerical_data(series: pd.Series) -> pd.Series:
        """Enhanced numerical data cleaning."""
        # Convert to string and clean
        cleaned = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        
        # Handle empty strings
        cleaned = cleaned.replace('', '0')
        
        # Convert to numeric
        numeric_series = pd.to_numeric(cleaned, errors='coerce').fillna(0)
        
        # Remove negative values (shouldn't have negative sales/units)
        numeric_series = numeric_series.clip(lower=0)
        
        return numeric_series
    
    @staticmethod
    def extract_product_info(upstream_path: str, is_uploaded: bool = False) -> Tuple[str, str]:
        """Enhanced product information extraction."""
        if not upstream_path:
            return 'N/A', 'N/A'
        
        try:
            # Load first few rows
            if is_uploaded:
                df_up_hd = pd.read_csv(upstream_path, nrows=5)
            else:
                if not os.path.exists(upstream_path):
                    return 'N/A', 'N/A'
                df_up_hd = pd.read_csv(upstream_path, nrows=5)
            
            sku, product = DataProcessor._parse_product_fields(df_up_hd)
            
            logger.info(f"Extracted product info - SKU: {sku}, Product: {product}")
            return sku, product
            
        except Exception as e:
            logger.warning(f"Could not extract product info: {str(e)}")
            return 'N/A', 'N/A'
    
    @staticmethod
    def _parse_product_fields(df: pd.DataFrame) -> Tuple[str, str]:
        """Parse SKU and product name from dataframe."""
        sku, product = 'N/A', 'N/A'
        
        # Search patterns
        sku_patterns = ['asin', 'sku', 'id', 'item', 'product_id']
        name_patterns = ['name', 'title', 'product', 'description', 'item_name']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for SKU patterns
            if any(pattern in col_lower for pattern in sku_patterns):
                if not any(exclude in col_lower for exclude in ['week', 'date', 'forecast']):
                    value = str(df[col].iloc[0]).strip()
                    if value and value != 'nan' and len(value) < 50:
                        sku = value
                        break
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for product name patterns
            if any(pattern in col_lower for pattern in name_patterns):
                if not any(exclude in col_lower for exclude in ['week', 'date', 'forecast']):
                    value = str(df[col].iloc[0]).strip()
                    if value and value != 'nan' and len(value) > 3:
                        product = value
                        break
        
        return sku, product

class ForecastingEngine:
    """Enhanced forecasting engine with multiple models."""
    
    def __init__(self):
        self.models = {
            'Linear Trend': self._linear_trend_forecast,
            'Exponential Smoothing': self._exponential_smoothing_forecast,
        }
        
        # Add advanced models if available
        if LIB_MANAGER.is_available('prophet'):
            self.models['Prophet'] = self._prophet_forecast
        if LIB_MANAGER.is_available('statsmodels'):
            self.models['ARIMA'] = self._arima_forecast
        if LIB_MANAGER.is_available('ml'):
            self.models['XGBoost'] = self._xgboost_forecast
    
    def generate_forecast(self, df_hist: pd.DataFrame, periods: int, model_choice: str, future_idx: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate forecast using selected model with error handling."""
        try:
            if len(df_hist) < CONFIG.MIN_HISTORICAL_WEEKS:
                st.warning(f"⚠️ Only {len(df_hist)} weeks of historical data. Minimum {CONFIG.MIN_HISTORICAL_WEEKS} recommended.")
            
            forecast_func = self.models.get(model_choice, self._exponential_smoothing_forecast)
            forecasts = forecast_func(df_hist, periods)
            
            # Ensure non-negative forecasts
            forecasts = np.maximum(forecasts, 0)
            
            df_fc = pd.DataFrame({
                'Week_Start': future_idx,
                'yhat': forecasts
            })
            
            logger.info(f"Generated {model_choice} forecast for {periods} periods")
            return df_fc
            
        except Exception as e:
            logger.error(f"Forecast generation failed with {model_choice}: {str(e)}")
            st.error(f"❌ Forecast failed: {str(e)}")
            
            # Fallback to simple method
            st.info("🔄 Falling back to Linear Trend forecast...")
            forecasts = self._linear_trend_forecast(df_hist, periods)
            return pd.DataFrame({
                'Week_Start': future_idx,
                'yhat': np.maximum(forecasts, 0)
            })
    
    def _linear_trend_forecast(self, df_hist: pd.DataFrame, periods: int) -> np.ndarray:
        """Simple linear trend forecast."""
        if len(df_hist) < 2:
            return np.full(periods, df_hist['y'].mean() if not df_hist.empty else 0)
        
        # Calculate trend
        x = np.arange(len(df_hist))
        y = df_hist['y'].values
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate forecast
        future_x = np.arange(len(df_hist), len(df_hist) + periods)
        forecasts = intercept + slope * future_x
        
        return forecasts
    
    def _exponential_smoothing_forecast(self, df_hist: pd.DataFrame, periods: int) -> np.ndarray:
        """Enhanced exponential smoothing with seasonality detection."""
        if len(df_hist) < 2:
            return np.full(periods, df_hist['y'].mean() if not df_hist.empty else 0)
        
        # Use statsmodels if available for better exponential smoothing
        if LIB_MANAGER.is_available('statsmodels'):
            try:
                ExponentialSmoothing = LIB_MANAGER.get_module('statsmodels', 'ExponentialSmoothing')
                
                # Detect seasonality
                seasonal = 'add' if len(df_hist) >= 24 else None
                seasonal_periods = 12 if seasonal else None
                
                model = ExponentialSmoothing(
                    df_hist['y'], 
                    trend='add',
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods
                )
                fitted = model.fit(optimized=True)
                forecasts = fitted.forecast(periods)
                return forecasts.values
            except:
                pass  # Fall back to simple method
        
        # Simple exponential smoothing implementation
        values = df_hist['y'].values
        alpha = 0.3  # Smoothing parameter
        beta = 0.1   # Trend parameter
        
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
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def _prophet_forecast(self, df_hist: pd.DataFrame, periods: int) -> np.ndarray:
        """Prophet forecast with enhanced configuration."""
        Prophet = LIB_MANAGER.get_module('prophet', 'Prophet')
        
        # Prepare data
        prophet_df = df_hist.rename(columns={'Week_Start': 'ds', 'y': 'y'})
        
        # Configure Prophet
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True if len(df_hist) > 52 else False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            uncertainty_samples=100
        )
        
        # Fit model
        m.fit(prophet_df)
        
        # Create future dataframe
        future = m.make_future_dataframe(periods=periods, freq='W')
        future = future.tail(periods)  # Only forecast periods
        
        # Generate forecast
        forecast = m.predict(future)
        
        return forecast['yhat'].values
    
    def _arima_forecast(self, df_hist: pd.DataFrame, periods: int) -> np.ndarray:
        """ARIMA forecast with automatic order selection."""
        ARIMA = LIB_MANAGER.get_module('statsmodels', 'ARIMA')
        
        if len(df_hist) < 10:
            return self._exponential_smoothing_forecast(df_hist, periods)
        
        try:
            # Simple auto ARIMA logic
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Try different ARIMA orders
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(df_hist['y'], order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit best model
            model = ARIMA(df_hist['y'], order=best_order)
            fitted = model.fit()
            forecast = fitted.get_forecast(steps=periods)
            
            return forecast.predicted_mean.values
            
        except Exception as e:
            logger.warning(f"ARIMA forecast failed: {str(e)}")
            return self._exponential_smoothing_forecast(df_hist, periods)
    
    def _xgboost_forecast(self, df_hist: pd.DataFrame, periods: int) -> np.ndarray:
        """Enhanced XGBoost forecast with feature engineering."""
        if len(df_hist) < 8:  # Need more data for ML
            return self._exponential_smoothing_forecast(df_hist, periods)
        
        try:
            xgb = LIB_MANAGER.get_module('ml', 'xgb')
            StandardScaler = LIB_MANAGER.get_module('ml', 'StandardScaler')
            
            # Feature engineering
            df_features = self._create_features(df_hist.copy())
            
            # Prepare training data
            feature_cols = [c for c in df_features.columns if c not in ['Week_Start', 'y']]
            X = df_features[feature_cols].fillna(0)
            y = df_features['y']
            
            if len(X) < 4:
                return self._exponential_smoothing_forecast(df_hist, periods)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                objective='reg:squarederror'
            )
            model.fit(X_scaled, y)
            
            # Generate forecasts
            forecasts = self._generate_ml_forecasts(model, scaler, df_features, feature_cols, periods)
            
            return np.array(forecasts)
            
        except Exception as e:
            logger.warning(f"XGBoost forecast failed: {str(e)}")
            return self._exponential_smoothing_forecast(df_hist, periods)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning models."""
        df = df.sort_values('Week_Start').copy()
        
        # Lag features
        for lag in [1, 2, 4, 8]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling statistics
        for window in [2, 4, 8]:
            if len(df) >= window:
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window, min_periods=1).std().fillna(0)
        
        # Time features
        df['week_of_year'] = df['Week_Start'].dt.isocalendar().week
        df['month'] = df['Week_Start'].dt.month
        df['quarter'] = df['Week_Start'].dt.quarter
        
        # Trend
        df['trend'] = range(len(df))
        
        return df.dropna()
    
    def _generate_ml_forecasts(self, model, scaler, df_features, feature_cols, periods):
        """Generate ML forecasts iteratively."""
        forecasts = []
        last_row = df_features.iloc[-1:].copy()
        
        for i in range(periods):
            # Prepare features
            X_pred = last_row[feature_cols].fillna(0)
            X_pred_scaled = scaler.transform(X_pred)
            
            # Predict
            pred = model.predict(X_pred_scaled)[0]
            forecasts.append(max(pred, 0))
            
            # Update features for next prediction
            last_row = last_row.copy()
            
            # Update lags
            if 'lag_1' in feature_cols:
                last_row['lag_1'] = forecasts[-1]
            if 'lag_2' in feature_cols and len(forecasts) >= 2:
                last_row['lag_2'] = forecasts[-2]
            
            # Update trend
            if 'trend' in feature_cols:
                last_row['trend'] = last_row['trend'].iloc[0] + 1
        
        return forecasts

class ReplenishmentCalculator:
    """Enhanced replenishment calculation with safety stock."""
    
    @staticmethod
    def calculate_replenishment(forecast_df: pd.DataFrame, init_inventory: int, 
                              woc_target: int, forecast_col: str, safety_factor: float = 1.0) -> pd.DataFrame:
        """Calculate replenishment with enhanced logic."""
        result_df = forecast_df.copy()
        
        on_hand = []
        replenishment = []
        weeks_of_cover = []
        safety_stock = []
        
        prev_on_hand = init_inventory
        
        for _, row in result_df.iterrows():
            demand = max(row[forecast_col], 0)
            
            # Calculate weeks of cover
            if demand > 0:
                woc = (prev_on_hand + replen_needed) / demand
                weeks_of_cover.append(round(woc, 2))
            else:
                weeks_of_cover.append(999.99)
            
            # Update on-hand for next iteration
            prev_on_hand = prev_on_hand + replen_needed - demand
        
        # Add calculated columns
        result_df['On_Hand_Begin'] = on_hand
        result_df['Replenishment'] = replenishment
        result_df['Safety_Stock'] = safety_stock
        result_df['Weeks_Of_Cover'] = weeks_of_cover
        result_df['On_Hand_End'] = [max(0, oh + rep - max(row[forecast_col], 0)) 
                                   for oh, rep, (_, row) in zip(on_hand, replenishment, result_df.iterrows())]
        
        return result_df

class AmazonForecastProcessor:
    """Enhanced Amazon forecast file processor."""
    
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def load_amazon_sellout_forecast(upstream_path: str, is_uploaded: bool = False) -> pd.DataFrame:
        """Load and parse Amazon sellout forecast with better error handling."""
        if not upstream_path:
            return pd.DataFrame()
        
        try:
            # Load data
            if is_uploaded:
                df_up = pd.read_csv(upstream_path, skiprows=1)
            else:
                if not os.path.exists(upstream_path):
                    return pd.DataFrame()
                df_up = pd.read_csv(upstream_path, skiprows=1)
            
            if df_up.empty:
                return pd.DataFrame()
            
            # Parse forecast data
            records = AmazonForecastProcessor._parse_forecast_columns(df_up)
            
            if not records:
                logger.warning("No valid forecast data found in Amazon file")
                return pd.DataFrame()
            
            result_df = pd.DataFrame(records)
            result_df = result_df.sort_values('Week_Start')
            
            logger.info(f"Loaded {len(result_df)} weeks of Amazon forecast data")
            return result_df
            
        except Exception as e:
            logger.warning(f"Could not load Amazon sellout forecast: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _parse_forecast_columns(df: pd.DataFrame) -> List[Dict]:
        """Parse forecast columns with multiple date formats."""
        records = []
        current_year = datetime.now().year
        
        # Date parsing patterns
        date_patterns = [
            (r"Week \d+.*?\((\d{1,2} [A-Za-z]+)", '%d %b'),  # Week X (DD MMM
            (r"(\d{1,2}/\d{1,2})", '%m/%d'),  # MM/DD format
            (r"(\d{1,2}-\d{1,2})", '%m-%d'),  # MM-DD format
        ]
        
        for col in df.columns:
            col_str = str(col)
            
            # Try each date pattern
            for pattern, date_format in date_patterns:
                date_match = re.search(pattern, col_str)
                if date_match:
                    try:
                        date_str = f"{date_match.group(1)}"
                        
                        # Add year if not present
                        if date_format in ['%d %b', '%m/%d', '%m-%d']:
                            date_str = f"{date_str} {current_year}"
                            if date_format == '%d %b':
                                date_format = '%d %b %Y'
                            elif date_format == '%m/%d':
                                date_format = '%m/%d/%Y'
                            else:
                                date_format = '%m-%d-%Y'
                        
                        parsed_date = pd.to_datetime(date_str, format=date_format, errors='coerce')
                        
                        if parsed_date is not None:
                            # Extract and clean value
                            raw_value = AmazonForecastProcessor._extract_numeric_value(df[col])
                            
                            if raw_value > 0:
                                records.append({
                                    'Week_Start': parsed_date,
                                    'Amazon_Sellout_Forecast': int(round(raw_value))
                                })
                                break  # Found valid date, move to next column
                    except Exception as e:
                        logger.debug(f"Date parsing failed for column {col}: {str(e)}")
                        continue
        
        return records
    
    @staticmethod
    def _extract_numeric_value(series: pd.Series) -> float:
        """Extract numeric value from series with enhanced cleaning."""
        if series.empty:
            return 0.0
        
        # Get first non-null value
        value = None
        for val in series:
            if pd.notna(val) and str(val).strip():
                value = val
                break
        
        if value is None:
            return 0.0
        
        # Clean and convert
        clean_value = str(value).replace(',', '').replace(' safety stock (percentage of demand)
            safety = demand * (safety_factor - 1.0)
            safety_stock.append(int(safety))
            
            # Target inventory = (demand * WOC) + safety stock
            target_inventory = (demand * woc_target) + safety
            
            # Replenishment needed
            replen_needed = max(target_inventory - prev_on_hand, 0)
            
            # Record values
            on_hand.append(int(prev_on_hand))
            replenishment.append(int(replen_needed))
            
            # Calculate, '')
        clean_value = re.sub(r'[^\d.-]', '', clean_value)
        
        try:
            return float(clean_value) if clean_value else 0.0
        except ValueError:
            return 0.0

class VisualizationEngine:
    """Enhanced visualization with better charts and insights."""
    
    def __init__(self):
        self.use_plotly = LIB_MANAGER.is_available('plotly')
    
    def create_forecast_dashboard(self, df_fc: pd.DataFrame, periods: int, 
                                projection_type: str, y_label: str, model_used: str):
        """Create comprehensive forecast dashboard."""
        
        # Header
        st.subheader(f"📊 {periods}-Week Replenishment Dashboard ({projection_type})")
        
        # Model info
        st.info(f"🤖 Forecast Model: **{model_used}**")
        
        # Key metrics
        self._display_key_metrics(df_fc, y_label)
        
        # Main forecast chart
        self._create_main_chart(df_fc, y_label)
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._create_inventory_chart(df_fc)
        
        with col2:
            self._create_woc_chart(df_fc)
    
    def _display_key_metrics(self, df_fc: pd.DataFrame, y_label: str):
        """Display key performance metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        # Total replenishment
        total_replen = df_fc['Replenishment'].sum()
        with col1:
            st.metric(
                "📦 Total Replenishment", 
                f"{total_replen:,} units",
                help="Total units to be replenished over forecast period"
            )
        
        # Average weekly demand
        avg_demand = df_fc.iloc[:, 1].mean()  # First data column after date
        with col2:
            st.metric(
                "📊 Avg Weekly Demand", 
                f"{avg_demand:,.0f} units",
                help="Average weekly demand over forecast period"
            )
        
        # Peak demand week
        peak_demand = df_fc.iloc[:, 1].max()
        with col3:
            st.metric(
                "📈 Peak Demand", 
                f"{peak_demand:,.0f} units",
                help="Highest weekly demand in forecast period"
            )
        
        # Average weeks of cover
        avg_woc = df_fc['Weeks_Of_Cover'].mean()
        with col4:
            st.metric(
                "📅 Avg Weeks of Cover", 
                f"{avg_woc:.1f} weeks",
                help="Average inventory coverage in weeks"
            )
    
    def _create_main_chart(self, df_fc: pd.DataFrame, y_label: str):
        """Create main forecast visualization."""
        metrics = ['Replenishment']
        forecast_col = df_fc.columns[1]  # First data column after date
        metrics.insert(0, forecast_col)
        
        if 'Amazon_Sellout_Forecast' in df_fc.columns:
            metrics.insert(1, 'Amazon_Sellout_Forecast')
        
        if self.use_plotly:
            self._create_plotly_main_chart(df_fc, metrics, y_label)
        else:
            st.line_chart(df_fc.set_index('Week_Start')[metrics])
    
    def _create_plotly_main_chart(self, df_fc: pd.DataFrame, metrics: List[str], y_label: str):
        """Create enhanced Plotly main chart."""
        px = LIB_MANAGER.get_module('plotly', 'px')
        go = LIB_MANAGER.get_module('plotly', 'go')
        
        fig = go.Figure()
        
        colors = ['#146EB4', '#FF9900', '#28a745']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=df_fc['Week_Start'],
                y=df_fc[metric],
                mode='lines+markers',
                name=metric.replace('_', ' '),
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{metric}</b><br>Date: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Demand Forecast vs Replenishment Plan',
            xaxis_title='Week',
            yaxis_title=y_label,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(tickformat='%d-%m-%Y')
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_inventory_chart(self, df_fc: pd.DataFrame):
        """Create inventory levels chart."""
        st.write("**📦 Inventory Levels**")
        
        inventory_data = df_fc[['Week_Start', 'On_Hand_Begin', 'On_Hand_End']].set_index('Week_Start')
        
        if self.use_plotly:
            px = LIB_MANAGER.get_module('plotly', 'px')
            fig = px.line(
                inventory_data,
                title='Inventory Levels',
                labels={'value': 'Units', 'variable': 'Inventory Type'}
            )
            fig.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(inventory_data)
    
    def _create_woc_chart(self, df_fc: pd.DataFrame):
        """Create weeks of cover chart."""
        st.write("**📅 Weeks of Cover**")
        
        woc_data = df_fc[['Week_Start', 'Weeks_Of_Cover']].set_index('Week_Start')
        
        if self.use_plotly:
            px = LIB_MANAGER.get_module('plotly', 'px')
            fig = px.line(
                woc_data,
                title='Weeks of Cover',
                labels={'value': 'Weeks', 'Week_Start': 'Date'}
            )
            fig.add_hline(y=4, line_dash="dash", line_color="red", 
                         annotation_text="Target WOC")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(woc_data)

def main():
    """Enhanced main application with better UX."""
    
    # Load custom CSS
    load_custom_css()
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1>🚀 Amazon Replenishment Forecast</h1>
        <p>Advanced inventory planning and demand forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    data_processor = DataProcessor()
    forecast_engine = ForecastingEngine()
    replen_calculator = ReplenishmentCalculator()
    amazon_processor = AmazonForecastProcessor()
    viz_engine = VisualizationEngine()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File uploads with validation
        st.subheader("📁 Data Files")
        sales_file = st.file_uploader(
            "📈 Sales History (CSV)", 
            type=["csv"],
            help="Upload historical sales data with dates and values"
        )
        
        if sales_file and not DataValidator.validate_file_size(sales_file):
            st.stop()
        
        fcst_file = st.file_uploader(
            "🔮 Amazon Forecast (CSV)", 
            type=["csv"],
            help="Optional: Amazon's sellout forecast for comparison"
        )
        
        if fcst_file and not DataValidator.validate_file_size(fcst_file):
            st.stop()
        
        # Model and parameters
        st.subheader("🤖 Model Settings")
        
        available_models = LIB_MANAGER.get_available_models()
        model_choice = st.selectbox(
            "Forecast Model", 
            available_models,
            help="Choose forecasting algorithm"
        )
        
        projection_type = st.selectbox(
            "Projection Type", 
            ["Units", "Sales $"],
            help="Forecast in units or dollar values"
        )
        
        # Inventory parameters
        st.subheader("📦 Inventory Settings")
        
        init_inv = st.number_input(
            "Current Inventory (units)", 
            min_value=0, 
            value=CONFIG.INIT_INVENTORY,
            help="Current on-hand inventory level"
        )
        
        woc_target = st.slider(
            "Target Weeks of Cover", 
            CONFIG.MIN_WOC, 
            CONFIG.MAX_WOC, 
            CONFIG.WOC_TARGET,
            help="Target inventory coverage in weeks"
        )
        
        safety_factor = st.slider(
            "Safety Stock Factor",
            1.0,
            2.0,
            1.1,
            0.1,
            help="Safety stock multiplier (1.0 = no safety stock)"
        )
        
        periods = st.number_input(
            "Forecast Horizon (weeks)", 
            min_value=CONFIG.MIN_PERIODS, 
            max_value=CONFIG.MAX_PERIODS, 
            value=CONFIG.PERIODS,
            help="Number of weeks to forecast"
        )
        
        # Validation
        is_valid, errors = DataValidator.validate_forecast_inputs(init_inv, woc_target, periods)
        if not is_valid:
            for error in errors:
                st.error(f"❌ {error}")
            st.stop()
        
        st.divider()
        
        # Library status
        with st.expander("📚 Available Models"):
            for lib, info in LIB_MANAGER.libraries.items():
                status = "✅" if info['available'] else "❌"
                st.write(f"{status} {lib.title()}")
        
        # Run button
        run_forecast = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    # Main content area
    if not run_forecast:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h2>🎯 Welcome to Advanced Forecasting</h2>
                <p>Upload your sales data and configure settings to generate accurate replenishment forecasts.</p>
                <br>
                <p><strong>Features:</strong></p>
                <ul style='text-align: left; display: inline-block;'>
                    <li>🤖 Multiple forecasting models</li>
                    <li>📊 Interactive visualizations</li>
                    <li>📈 Safety stock calculations</li>
                    <li>🔍 Amazon forecast comparison</li>
                    <li>📋 Detailed replenishment plans</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.stop()
    
    # Process forecast
    try:
        # Determine file paths
        sales_path = sales_file if sales_file else CONFIG.DEFAULT_SALES_PATH
        upstream_path = fcst_file if fcst_file else CONFIG.DEFAULT_UPSTREAM_PATH
        
        if not sales_path:
            st.error("❌ Sales history file is required")
            st.stop()
        
        # Load and process data
        with st.spinner('📊 Processing sales data...'):
            df_raw = data_processor.load_and_clean_sales_data(
                sales_path, is_uploaded=bool(sales_file)
            )
            
            # Extract product info
            sku, product = data_processor.extract_product_info(
                upstream_path, is_uploaded=bool(fcst_file)
            )
        
        # Product information display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📦 Product</h4>
                <p>{product}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏷️ SKU/ASIN</h4>
                <p>{sku}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Process historical data
        with st.spinner('🔍 Analyzing historical patterns...'):
            y_col, forecast_label, y_label = data_processor.detect_column_type(df_raw, projection_type)
            
            df_hist = pd.DataFrame({
                'Week_Start': df_raw['Week_Start'],
                'y': data_processor.clean_numerical_data(df_raw[y_col])
            })
            
            # Filter and clean
            df_hist = df_hist[df_hist['Week_Start'] <= pd.to_datetime(datetime.now().date())]
            df_hist = df_hist.drop_duplicates(subset=['Week_Start']).sort_values('Week_Start')
            
            if df_hist.empty:
                st.error("❌ No valid historical data found")
                st.stop()
        
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
            df_fc = forecast_engine.generate_forecast(df_hist, periods, model_choice, future_idx)
            df_fc[forecast_label] = df_fc['yhat'].round(0).astype(int)
        
        # Calculate replenishment
        with st.spinner('📊 Calculating replenishment plan...'):
            df_fc = replen_calculator.calculate_replenishment(
                df_fc, init_inv, woc_target, forecast_label, safety_factor
            )
        
        # Load Amazon forecast if available
        if upstream_path:
            with st.spinner('🔄 Loading Amazon forecast...'):
                amazon_forecast = amazon_processor.load_amazon_sellout_forecast(
                    upstream_path, is_uploaded=bool(fcst_file)
                )
                if not amazon_forecast.empty:
                    df_fc = df_fc.merge(amazon_forecast, on='Week_Start', how='left')
                    st.success("✅ Amazon forecast data loaded successfully")
        
        # Format for display
        df_fc['Date'] = df_fc['Week_Start'].dt.strftime('%d-%m-%Y')
        
        # Create visualizations
        viz_engine.create_forecast_dashboard(df_fc, periods, projection_type, y_label, model_choice)
        
        # Detailed results table
        st.subheader("📋 Detailed Replenishment Plan")
        
        # Prepare display columns
        display_columns = ['Date', forecast_label, 'On_Hand_Begin', 'Replenishment', 
                          'Safety_Stock', 'On_Hand_End', 'Weeks_Of_Cover']
        
        if 'Amazon_Sellout_Forecast' in df_fc.columns:
            display_columns.insert(1, 'Amazon_Sellout_Forecast')
        
        # Format numbers for better display
        display_df = df_fc[display_columns].copy()
        for col in display_df.columns:
            if col not in ['Date'] and display_df[col].dtype in ['int64', 'float64']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df_fc.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"replenishment_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Success message
        st.success(f"✅ Forecast generated successfully using {model_choice} model!")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please check your data files and try again.")

if __name__ == "__main__":
    main() safety stock (percentage of demand)
            safety = demand * (safety_factor - 1.0)
            safety_stock.append(int(safety))
            
            # Target inventory = (demand * WOC) + safety stock
            target_inventory = (demand * woc_target) + safety
            
            # Replenishment needed
            replen_needed = max(target_inventory - prev_on_hand, 0)
            
            # Record values
            on_hand.append(int(prev_on_hand))
            replenishment.append(int(replen_needed))
            
            # Calculate
