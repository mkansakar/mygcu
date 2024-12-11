import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def calculate_bollinger_bands(data, column='Close', window=20):
    """
    Calculate Bollinger Bands for a given column.
    """
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    return rolling_mean, upper_band, lower_band

def calculate_atr(data, window=14):
    """
    Calculate Average True Range (ATR).
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def display_volatility_indicators():
    """
    Streamlit interface for displaying volatility indicators.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("No data found. Please load data first.")
        return

    st.title("Volatility Indicators")

    # Select column and window size
    data = st.session_state['data']
    st.subheader("Bollinger Bands")
    window_bb = st.slider("Select Window Size for Bollinger Bands", min_value=5, max_value=50, value=20, step=1)

    # Calculate Bollinger Bands
    rolling_mean, upper_band, lower_band = calculate_bollinger_bands(data, window=window_bb)

    # Plot Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=rolling_mean, mode='lines', name=f'{window_bb}-Day SMA', line=dict(color='orange')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=upper_band, mode='lines', name='Upper Band', line=dict(dash='dot', color='green')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=lower_band, mode='lines', name='Lower Band', line=dict(dash='dot', color='red')))
    fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig_bb)

    # Average True Range (ATR)
    st.subheader("Average True Range (ATR)")
    window_atr = st.slider("Select Window Size for ATR", min_value=5, max_value=50, value=14, step=1)

    # Calculate and display ATR
    atr = calculate_atr(data, window=window_atr)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=data.index, y=atr, mode='lines', name='ATR', line=dict(color='purple')))
    fig_atr.update_layout(title=f"Average True Range (ATR) - {window_atr} Day Window", xaxis_title="Date", yaxis_title="ATR", template="plotly_white")
    st.plotly_chart(fig_atr)

    # Standard Deviation
    st.subheader("Standard Deviation of Prices")
    window_std = st.slider("Select Window Size for Standard Deviation", min_value=5, max_value=50, value=20, step=1)

    # Calculate and display Standard Deviation
    rolling_std = data['Close'].rolling(window=window_std).std()
    fig_std = go.Figure()
    fig_std.add_trace(go.Scatter(x=data.index, y=rolling_std, mode='lines', name='Standard Deviation', line=dict(color='blue')))
    fig_std.update_layout(title=f"Standard Deviation - {window_std} Day Window", xaxis_title="Date", yaxis_title="Standard Deviation", template="plotly_white")
    st.plotly_chart(fig_std)
