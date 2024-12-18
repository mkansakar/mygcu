import streamlit as st
import pandas as pd
import numpy as np

def calculate_rsi(data, column='Close', window=14):
    """
    Calculate the Relative Strength Index (RSI) without modifying the original dataset.
    """
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD and Signal Line without modifying the original dataset.
    """
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_moving_averages(data, column='Close', windows=[5, 15, 30]):
    """
    Calculate moving averages for the specified windows without modifying the original dataset.
    """
    return {f"SMA_{window}": data[column].rolling(window=window).mean() for window in windows}

def moving_indicators():
    """
    Display moving averages, RSI, and MACD plots without modifying the original dataset.
    """
    st.title("Moving Averages and Indicators")

    if 'data' not in st.session_state:
        st.error("Please load the data first from the sidebar on the left.")
        return

    data = st.session_state['data']

    st.header("Moving Averages")
    windows = st.multiselect("Select Moving Average Periods", [5, 15, 30], default=[5, 15, 30])
    moving_avg = calculate_moving_averages(data, windows=windows)

    for period, avg in moving_avg.items():
        st.subheader(f"{period} - Moving Average")
        st.line_chart(pd.DataFrame({period: avg}, index=data.index))

    st.header("Relative Strength Index (RSI)")
    rsi_window = st.slider("Select RSI Window", 10, 30, 14)
    rsi = calculate_rsi(data, window=rsi_window)
    st.line_chart(pd.DataFrame({"RSI": rsi}, index=data.index))

    st.header("MACD")
    short_window = st.slider("Short EMA Window", 5, 20, 12)
    long_window = st.slider("Long EMA Window", 20, 50, 26)
    signal_window = st.slider("Signal Line Window", 5, 20, 9)
    macd, signal_line = calculate_macd(data, short_window=short_window, long_window=long_window, signal_window=signal_window)
    macd_df = pd.DataFrame({"MACD": macd, "Signal Line": signal_line}, index=data.index)
    st.line_chart(macd_df)

    st.info("Note: These plots are based on calculated values and do not modify the original dataset.")
