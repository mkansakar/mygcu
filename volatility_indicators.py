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
        st.error("Please load the data first from the sidebar on the left.")
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
    with st.expander("What is Moving Bands?"):
        st.write("""
            Bollinger Bands are a technical analysis tool used in stock price forecasting to measure price volatility and identify potential buy or sell signals.\n     
            When the price touches or exceeds the upper band, it suggests the stock may be overbought, signaling a potential reversal or price drop.\n
            When the price touches or falls below the lower band, it indicates the stock may be oversold, signaling a potential reversal or price increase.\n
            Narrow bands indicate low volatility, often preceding significant price movements or breakouts.\n
            Wide bands indicate high volatility, reflecting a period of significant price fluctuations.\n
            Prices consistently hugging the upper band during an uptrend may signal strong bullish momentum.\n
            Prices consistently hugging the lower band during a downtrend may indicate sustained bearish momentum.\n
        """)  
    # Average True Range (ATR)
    st.subheader("Average True Range (ATR)")
    window_atr = st.slider("Select Window Size for ATR", min_value=5, max_value=50, value=14, step=1)

    # Calculate and display ATR
    atr = calculate_atr(data, window=window_atr)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=data.index, y=atr, mode='lines', name='ATR', line=dict(color='purple')))
    fig_atr.update_layout(title=f"Average True Range (ATR) - {window_atr} Day Window", xaxis_title="Date", yaxis_title="ATR", template="plotly_white")
    st.plotly_chart(fig_atr)
    with st.expander("What is ATR?"):
        st.write("""
            Average True Range (ATR) is a technical analysis indicator used to measure the volatility of a stock's price.\n     
            High ATR: Indicates increased volatility, meaning the stock experiences large price movements within a given timeframe. Often observed during market news, earnings announcements, or significant economic events. Higher ATR suggests greater risk and reward potential for traders.\n
            Low ATR: Indicates low volatility, where the stock price moves within a relatively narrow range. Often occurs during periods of consolidation or when the market lacks significant catalysts. Lower ATR implies lower risk but also limited reward potential.\n
            Trend Confirmation: A rising ATR during a price uptrend or downtrend confirms strong momentum, suggesting the trend may continue. A declining ATR during a trend could signal weakening momentum and a potential reversal or consolidation.\n
            Breakout Signals: An increase in ATR, especially after a prolonged period of low ATR, may indicate the beginning of a new trend or breakout.\n
        """)  


    # Standard Deviation
    st.subheader("Standard Deviation of Prices")
    window_std = st.slider("Select Window Size for Standard Deviation", min_value=5, max_value=50, value=20, step=1)

    # Calculate and display Standard Deviation
    rolling_std = data['Close'].rolling(window=window_std).std()
    fig_std = go.Figure()
    fig_std.add_trace(go.Scatter(x=data.index, y=rolling_std, mode='lines', name='Standard Deviation', line=dict(color='blue')))
    fig_std.update_layout(title=f"Standard Deviation - {window_std} Day Window", xaxis_title="Date", yaxis_title="Standard Deviation", template="plotly_white")
    st.plotly_chart(fig_std)
    with st.expander("What is Standard Deviation of Prices?"):
        st.write("""
        Standard deviation in stock price forecasting is a statistical measure that quantifies the amount of variation or dispersion of stock price data over a specific period.\n
        Low Standard Deviation: Indicates that stock prices are closely clustered around the mean, suggesting low volatility. Implies that the stock's price is relatively stable.\n 
        High Standard Deviation: Suggests that stock prices are spread out widely from the mean, indicating high volatility. Reflects a greater level of uncertainty and risk in price movements.\n
        Changes in Standard Deviation: An increasing standard deviation may signal the onset of higher market activity or significant news events affecting the stock. A decreasing standard deviation often occurs during periods of consolidation or low market activity.                 
    """)
