#data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

def calculate_rsi(data, column='Close', window=14):
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_atr(data, high='High', low='Low', close='Close', window=14):
    high_low = data[high] - data[low]
    high_close = (data[high] - data[close].shift()).abs()
    low_close = (data[low] - data[close].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_cmf(data, high='High', low='Low', close='Close', volume='Volume', window=20):
    money_flow_multiplier = ((data[close] - data[low]) - (data[high] - data[close])) / (data[high] - data[low])
    money_flow_volume = money_flow_multiplier * data[volume]
    cmf = money_flow_volume.rolling(window=window).sum() / data[volume].rolling(window=window).sum()
    return cmf


def preprocess_data(data):
    data = data.copy()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['ATR'] = calculate_atr(data)
    data['CMF'] = calculate_cmf(data)
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data['Daily_Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

def split_data(data):
    features = data[['Close', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Signal_Line', 'Momentum', 'Daily_Return', 'ATR', 'CMF', 'Gold_Close', 'GBPUSD']]
    target = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    tscv = TimeSeriesSplit(n_splits=5)
    splits = [(train_idx, test_idx) for train_idx, test_idx in tscv.split(features_scaled)]
    
    return features_scaled, target, splits