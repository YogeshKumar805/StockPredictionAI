import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def fetch_stock_data(symbol, period='2y', interval='1d'):
    """Fetch stock data from Yahoo Finance with multiple timeframe support"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        return df
    except Exception as e:
        raise Exception(f"Error fetching data for {symbol}: {str(e)}")

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_stochastic(data, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k_line = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    d_line = k_line.rolling(window=d_window).mean()
    return k_line, d_line

def prepare_data(df):
    """Prepare data for ML model with enhanced technical indicators"""
    df = df.copy()

    # Basic price indicators
    df['MA5'] = calculate_sma(df['Close'], 5)
    df['MA20'] = calculate_sma(df['Close'], 20)
    df['MA50'] = calculate_sma(df['Close'], 50)
    df['EMA12'] = calculate_ema(df['Close'], 12)
    df['EMA26'] = calculate_ema(df['Close'], 26)

    # Technical indicators
    df['RSI'] = calculate_rsi(df['Close'])
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    k_line, d_line = calculate_stochastic(df)
    df['Stoch_K'] = k_line
    df['Stoch_D'] = d_line

    # Volatility and momentum
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['MOM'] = df['Close'].diff(periods=10)

    # Target variable
    df['Target'] = df['Close'].shift(-1)

    # Feature selection
    features = [
        'Close', 'Volume', 'MA5', 'MA20', 'MA50', 'EMA12', 'EMA26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Stoch_K', 'Stoch_D', 'Volatility', 'MOM'
    ]

    # Remove NaN values
    df = df.dropna()

    X = df[features]
    y = df['Target']

    return X, y, df

def calculate_metrics(y_true, y_pred):
    """Calculate prediction performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 4)
    }