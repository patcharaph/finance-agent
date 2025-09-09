"""
Finance Analysis Tools
=====================

This module contains all the tools needed for finance analysis:
- DataLoader: Fetch price data, news, and market data
- IndicatorCalculator: Calculate technical indicators
- SentimentAnalyzer: Analyze news sentiment
- ModelOps: Machine learning model operations
"""

import os
import time
import json
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import ta

# Optional import for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None

warnings.filterwarnings("ignore")


@dataclass
class DataResult:
    """Result container for data operations"""
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DataLoader:
    """Data loading and fetching tools"""
    
    def __init__(self, proxy_config: Optional[Dict[str, str]] = None):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.proxy_config = proxy_config or {}
        
        # Configure yfinance with proxy if provided
        if self.proxy_config:
            print(f"Info: Using proxy configuration: {self.proxy_config}")
            # Note: yfinance uses requests internally, proxy will be handled by requests
    
    def set_proxy(self, proxy_config: Dict[str, str]):
        """
        Set proxy configuration for network requests
        
        Args:
            proxy_config: Dictionary with proxy settings
                         Example: {"http": "http://proxy:port", "https": "https://proxy:port"}
        """
        self.proxy_config = proxy_config
        print(f"Info: Updated proxy configuration: {self.proxy_config}")
    
    def clear_proxy(self):
        """Clear proxy configuration"""
        self.proxy_config = {}
        print("Info: Cleared proxy configuration")
    
    def fetch_price_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> DataResult:
        """
        Fetch price data from Yahoo Finance with robust error handling
        
        Args:
            symbol: Stock symbol 
                   - Thai stocks: Use .BK suffix (e.g., "PTT.BK", "KBANK.BK", "SCB.BK")
                   - US stocks: Use direct symbols (e.g., "AAPL", "MSFT", "TSLA", "GOOGL")
                   - Indices: Use standard format (e.g., "^SETI", "^GSPC")
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")
                   - For latest day only: Use "5d" then tail(1) to avoid weekend/holiday issues
            interval: Data interval ("1d", "1h", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
        
        Returns:
            DataResult with price data
        """
        try:
            # Check cache first
            cache_key = f"price_{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return DataResult(True, cached_data, metadata={"cached": True})
            
            # Validate symbol format
            if not symbol or not isinstance(symbol, str):
                return DataResult(False, error=f"Invalid symbol format: {symbol}")
            
            # Try to fetch data with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Configure proxy for yfinance if available
                    if self.proxy_config:
                        import requests
                        session = requests.Session()
                        session.proxies.update(self.proxy_config)
                        ticker = yf.Ticker(symbol, session=session)
                    else:
                        ticker = yf.Ticker(symbol)
                    
                    df = ticker.history(period=period, interval=interval, auto_adjust=True)
                    
                    if df is None or df.empty:
                        # Check if this is a connectivity issue
                        if attempt < max_retries - 1:
                            print(f"Warning: No data for {symbol}, retrying... (attempt {attempt + 1})")
                            time.sleep(1)  # Wait before retry
                            continue
                        else:
                            # Try to generate mock data as fallback
                            print(f"Warning: No data found for {symbol}. Generating mock data for testing...")
                            mock_result = self.generate_mock_data(symbol, period, interval)
                            if mock_result.success:
                                return mock_result
                            else:
                                return DataResult(False, error=f"No data found for symbol: {symbol}. This could be due to network issues or invalid symbol.")
                    
                    # Standardize column names
                    df.columns = [col.lower() for col in df.columns]
                    df = df.dropna()
                    
                    # Handle special case for latest day only (avoid weekend/holiday issues)
                    if period == "5d" and len(df) > 1:
                        # Get the most recent trading day
                        df = df.tail(1)
                        print(f"Info: Using latest trading day only for {symbol} (avoiding weekends/holidays)")
                    
                    # Cache the result
                    self.cache[cache_key] = (df, time.time())
                    
                    metadata = {
                        "symbol": symbol,
                        "period": period,
                        "interval": interval,
                        "rows": len(df),
                        "start_date": str(df.index[0].date()),
                        "end_date": str(df.index[-1].date()),
                        "cached": False
                    }
                    
                    return DataResult(True, df, metadata=metadata)
                    
                except Exception as retry_error:
                    error_msg = str(retry_error)
                    error_type = type(retry_error).__name__
                    
                    # Enhanced error logging
                    print(f"Error fetching {symbol} (attempt {attempt + 1}/{max_retries}):")
                    print(f"  Error Type: {error_type}")
                    print(f"  Error Message: {error_msg}")
                    
                    # Check for specific error types
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        print(f"  Detected: Rate limit exceeded for {symbol}")
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5  # Exponential backoff
                            print(f"  Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                    elif "403" in error_msg or "forbidden" in error_msg.lower():
                        print(f"  Detected: Access forbidden for {symbol} - possible network filtering/proxy needed")
                    elif "timeout" in error_msg.lower():
                        print(f"  Detected: Network timeout for {symbol}")
                    elif "connection" in error_msg.lower():
                        print(f"  Detected: Connection error for {symbol}")
                    
                    if attempt < max_retries - 1:
                        print(f"  Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        # If all retries failed, try to provide helpful error message
                        if "JSONDecodeError" in error_msg or "Expecting value" in error_msg:
                            # Try to generate mock data as fallback
                            print(f"Warning: Yahoo Finance API unavailable for {symbol}. Generating mock data for testing...")
                            mock_result = self.generate_mock_data(symbol, period, interval)
                            if mock_result.success:
                                return mock_result
                            else:
                                return DataResult(False, error=f"Network/API error for {symbol}. Yahoo Finance may be temporarily unavailable. Please check your internet connection and try again later.")
                        else:
                            return DataResult(False, error=f"Error fetching price data for {symbol}: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            if "JSONDecodeError" in error_msg or "Expecting value" in error_msg:
                # Try to generate mock data as fallback
                print(f"Warning: Yahoo Finance API unavailable for {symbol}. Generating mock data for testing...")
                mock_result = self.generate_mock_data(symbol, period, interval)
                if mock_result.success:
                    return mock_result
                else:
                    return DataResult(False, error=f"Network/API error for {symbol}. Yahoo Finance may be temporarily unavailable. Please check your internet connection and try again later.")
            else:
                return DataResult(False, error=f"Error fetching price data: {error_msg}")
    
    def generate_mock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> DataResult:
        """
        Generate mock price data for testing when Yahoo Finance is unavailable
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
        
        Returns:
            DataResult with mock price data
        """
        try:
            # Calculate number of days based on period
            period_days = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 3650
            }
            days = period_days.get(period, 730)  # Default to 2y
            
            # Generate date range
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate realistic mock data
            np.random.seed(42)  # For reproducible results
            
            # Base price around 100
            base_price = 100.0
            if "PTT" in symbol.upper():
                base_price = 35.0  # PTT-like price
            elif "SET" in symbol.upper():
                base_price = 1500.0  # SET Index-like price
            
            # Generate price series with some trend and volatility
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            # Generate OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Generate realistic OHLC from close price
                volatility = 0.01
                high = price * (1 + np.random.uniform(0, volatility))
                low = price * (1 - np.random.uniform(0, volatility))
                open_price = prices[i-1] if i > 0 else price
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            df.columns = [col.lower() for col in df.columns]
            
            metadata = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "rows": len(df),
                "start_date": str(df.index[0].date()),
                "end_date": str(df.index[-1].date()),
                "cached": False,
                "mock_data": True,
                "note": "Generated mock data due to API unavailability"
            }
            
            return DataResult(True, df, metadata=metadata)
            
        except Exception as e:
            return DataResult(False, error=f"Error generating mock data: {str(e)}")
    
    def fetch_news_sentiment(self, symbol: str, days_back: int = 7) -> DataResult:
        """
        Fetch news and calculate sentiment (simplified version)
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back for news
        
        Returns:
            DataResult with sentiment data
        """
        try:
            # This is a simplified implementation
            # In production, you'd use proper news APIs like NewsAPI, Alpha Vantage, etc.
            
            # For demo purposes, we'll create mock sentiment data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days_back, freq='D')
            sentiment_scores = np.random.normal(0, 0.3, days_back)  # Mock sentiment
            
            df = pd.DataFrame({
                'date': dates,
                'sentiment_score': sentiment_scores,
                'news_count': np.random.randint(5, 20, days_back)
            })
            df.set_index('date', inplace=True)
            
            metadata = {
                "symbol": symbol,
                "days_back": days_back,
                "avg_sentiment": float(df['sentiment_score'].mean()),
                "total_news": int(df['news_count'].sum())
            }
            
            return DataResult(True, df, metadata=metadata)
            
        except Exception as e:
            return DataResult(False, error=f"Error fetching news sentiment: {str(e)}")
    
    def fetch_market_data(self, symbols: List[str], period: str = "1y") -> DataResult:
        """
        Fetch multiple symbols' data for market analysis
        
        Args:
            symbols: List of stock symbols
            period: Time period
        
        Returns:
            DataResult with combined market data
        """
        try:
            all_data = {}
            for symbol in symbols:
                result = self.fetch_price_data(symbol, period)
                if result.success:
                    all_data[symbol] = result.data['close']
                else:
                    print(f"Warning: Failed to fetch {symbol}: {result.error}")
            
            if not all_data:
                return DataResult(False, error="No data fetched for any symbol")
            
            # Combine all close prices
            combined_df = pd.DataFrame(all_data)
            combined_df = combined_df.dropna()
            
            metadata = {
                "symbols": symbols,
                "period": period,
                "successful_symbols": list(all_data.keys()),
                "rows": len(combined_df)
            }
            
            return DataResult(True, combined_df, metadata=metadata)
            
        except Exception as e:
            return DataResult(False, error=f"Error fetching market data: {str(e)}")


class IndicatorCalculator:
    """Technical indicator calculation tools"""
    
    def __init__(self):
        self.available_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic',
            'atr', 'adx', 'cci', 'williams_r', 'obv', 'volume_sma'
        ]
    
    def calculate_indicators(self, price_data: pd.DataFrame, indicators: List[str] = None) -> DataResult:
        """
        Calculate technical indicators for price data
        
        Args:
            price_data: DataFrame with OHLCV data
            indicators: List of indicators to calculate (None = all)
        
        Returns:
            DataResult with indicators
        """
        try:
            if indicators is None:
                indicators = self.available_indicators
            
            df = price_data.copy()
            calculated_indicators = []
            
            for indicator in indicators:
                if indicator not in self.available_indicators:
                    print(f"Warning: Unknown indicator: {indicator}")
                    continue
                
                try:
                    if indicator == 'sma':
                        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
                        calculated_indicators.extend(['sma_20', 'sma_50'])
                    
                    elif indicator == 'ema':
                        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
                        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
                        calculated_indicators.extend(['ema_12', 'ema_26'])
                    
                    elif indicator == 'rsi':
                        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
                        calculated_indicators.append('rsi_14')
                    
                    elif indicator == 'macd':
                        macd_line = ta.trend.macd(df['close'])
                        macd_signal = ta.trend.macd_signal(df['close'])
                        macd_hist = ta.trend.macd_diff(df['close'])
                        df['macd'] = macd_line
                        df['macd_signal'] = macd_signal
                        df['macd_histogram'] = macd_hist
                        calculated_indicators.extend(['macd', 'macd_signal', 'macd_histogram'])
                    
                    elif indicator == 'bollinger':
                        bb_high = ta.volatility.bollinger_hband(df['close'])
                        bb_low = ta.volatility.bollinger_lband(df['close'])
                        bb_mid = ta.volatility.bollinger_mavg(df['close'])
                        df['bb_high'] = bb_high
                        df['bb_low'] = bb_low
                        df['bb_mid'] = bb_mid
                        calculated_indicators.extend(['bb_high', 'bb_low', 'bb_mid'])
                    
                    elif indicator == 'stochastic':
                        stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'])
                        stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
                        df['stoch_k'] = stoch_k
                        df['stoch_d'] = stoch_d
                        calculated_indicators.extend(['stoch_k', 'stoch_d'])
                    
                    elif indicator == 'atr':
                        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
                        calculated_indicators.append('atr')
                    
                    elif indicator == 'adx':
                        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
                        calculated_indicators.append('adx')
                    
                    elif indicator == 'cci':
                        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
                        calculated_indicators.append('cci')
                    
                    elif indicator == 'williams_r':
                        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
                        calculated_indicators.append('williams_r')
                    
                    elif indicator == 'obv':
                        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                        calculated_indicators.append('obv')
                    
                    elif indicator == 'volume_sma':
                        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
                        calculated_indicators.append('volume_sma')
                
                except Exception as e:
                    print(f"Warning: Failed to calculate {indicator}: {str(e)}")
                    continue
            
            # Add basic features
            df['returns_1d'] = df['close'].pct_change()
            df['returns_5d'] = df['close'].pct_change(5)
            df['volatility_20d'] = df['returns_1d'].rolling(20).std()
            calculated_indicators.extend(['returns_1d', 'returns_5d', 'volatility_20d'])
            
            # Remove rows with NaN values
            df = df.dropna()
            
            metadata = {
                "requested_indicators": indicators,
                "calculated_indicators": calculated_indicators,
                "rows_after_calculation": len(df),
                "columns_added": len(calculated_indicators)
            }
            
            return DataResult(True, df, metadata=metadata)
            
        except Exception as e:
            return DataResult(False, error=f"Error calculating indicators: {str(e)}")
    
    def create_features(self, data: pd.DataFrame, target_horizon: int = 5) -> DataResult:
        """
        Create feature matrix for machine learning
        
        Args:
            data: DataFrame with price data and indicators
            target_horizon: Days ahead for target variable
        
        Returns:
            DataResult with features and target
        """
        try:
            df = data.copy()
            
            # Create target variable (future returns)
            df['target'] = df['close'].pct_change(target_horizon).shift(-target_horizon)
            
            # Select feature columns (exclude OHLCV and target)
            feature_cols = [col for col in df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
            
            # Create feature matrix
            features_df = df[feature_cols + ['target']].dropna()
            
            X = features_df[feature_cols]
            y = features_df['target']
            
            metadata = {
                "target_horizon": target_horizon,
                "feature_count": len(feature_cols),
                "feature_names": feature_cols,
                "samples": len(X),
                "target_mean": float(y.mean()),
                "target_std": float(y.std())
            }
            
            return DataResult(True, {"X": X, "y": y}, metadata=metadata)
            
        except Exception as e:
            return DataResult(False, error=f"Error creating features: {str(e)}")


class SentimentAnalyzer:
    """News and sentiment analysis tools"""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if not VADER_AVAILABLE or self.analyzer is None:
            # Return neutral sentiment if VADER is not available
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def analyze_news_batch(self, news_data: pd.DataFrame) -> DataResult:
        """
        Analyze sentiment for a batch of news
        
        Args:
            news_data: DataFrame with news text
        
        Returns:
            DataResult with sentiment analysis
        """
        try:
            if 'text' not in news_data.columns:
                return DataResult(False, error="No 'text' column found in news data")
            
            sentiments = []
            for text in news_data['text']:
                sentiment = self.analyze_text_sentiment(str(text))
                sentiments.append(sentiment)
            
            sentiment_df = pd.DataFrame(sentiments)
            result_df = pd.concat([news_data.reset_index(drop=True), sentiment_df], axis=1)
            
            metadata = {
                "news_count": len(news_data),
                "avg_compound": float(sentiment_df['compound'].mean()),
                "avg_positive": float(sentiment_df['positive'].mean()),
                "avg_negative": float(sentiment_df['negative'].mean()),
                "avg_neutral": float(sentiment_df['neutral'].mean())
            }
            
            return DataResult(True, result_df, metadata=metadata)
            
        except Exception as e:
            return DataResult(False, error=f"Error analyzing news batch: {str(e)}")


class ModelOps:
    """Machine learning model operations"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'linear_regression': LinearRegression
        }
        self.scaler = StandardScaler()
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest', 
                   test_size: float = 0.2, random_state: int = 42, **kwargs) -> DataResult:
        """
        Train a machine learning model
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train
            test_size: Proportion of data for testing
            random_state: Random seed
            **kwargs: Additional model parameters
        
        Returns:
            DataResult with trained model and metrics
        """
        try:
            if model_type not in self.models:
                return DataResult(False, error=f"Unknown model type: {model_type}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Set default parameters
            default_params = {
                'random_forest': {'n_estimators': 100, 'random_state': random_state},
                'gradient_boosting': {'random_state': random_state},
                'linear_regression': {}
            }
            
            # Merge with user parameters
            params = {**default_params[model_type], **kwargs}
            
            # Train model
            model_class = self.models[model_type]
            model = model_class(**params)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate naive baseline (using previous value)
            y_naive = np.roll(y_test.values, 1)
            y_naive[0] = y_test.iloc[0]  # Use first actual value
            mae_naive = mean_absolute_error(y_test, y_naive)
            
            # Relative performance
            rel_performance = mae / (mae_naive + 1e-9)
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'r2': float(r2),
                'mae_naive': float(mae_naive),
                'rel_performance': float(rel_performance),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            result_data = {
                'model': model,
                'scaler': self.scaler,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'metrics': metrics
            }
            
            return DataResult(True, result_data, metadata=metrics)
            
        except Exception as e:
            return DataResult(False, error=f"Error training model: {str(e)}")
    
    def predict(self, model, scaler, X: pd.DataFrame) -> DataResult:
        """
        Make predictions using trained model
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            X: Feature matrix
        
        Returns:
            DataResult with predictions
        """
        try:
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            return DataResult(True, predictions, metadata={"predictions_count": len(predictions)})
            
        except Exception as e:
            return DataResult(False, error=f"Error making predictions: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Test the tools
    print("Testing Finance Analysis Tools...")
    
    # Test DataLoader
    loader = DataLoader()
    result = loader.fetch_price_data("PTT.BK", period="1y")
    if result.success:
        print(f"✅ DataLoader: Fetched {len(result.data)} rows of data")
    else:
        print(f"❌ DataLoader: {result.error}")
    
    # Test IndicatorCalculator
    if result.success:
        calc = IndicatorCalculator()
        indicators_result = calc.calculate_indicators(result.data, ['rsi', 'sma', 'macd'])
        if indicators_result.success:
            print(f"✅ IndicatorCalculator: Calculated {len(indicators_result.metadata['calculated_indicators'])} indicators")
        else:
            print(f"❌ IndicatorCalculator: {indicators_result.error}")
    
    print("Tools testing completed!")
