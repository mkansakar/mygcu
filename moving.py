import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) and signal line.
    """
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def display_moving_analysis():
    """
    Display moving averages, RSI, and MACD for stock data.
    """
    if 'data' in st.session_state:
        st.title("Technical Indicators: Moving Averages, RSI, and MACD")

        # Load data
        data = st.session_state['data']

        # Moving Average
        st.subheader("Moving Averages")
        ma_period = st.selectbox("Select Moving Average Period", [5, 15, 30], key="ma_period")
        data[f"SMA_{ma_period}"] = data['Close'].rolling(window=ma_period).mean()

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data[f"SMA_{ma_period}"], mode='lines', name=f"{ma_period}-Day SMA"))
        fig_ma.update_layout(title="Moving Average", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_ma)

        # RSI
        st.subheader("Relative Strength Index (RSI)")
        rsi_period = st.slider("Select RSI Period", 7, 30, 14, key="rsi_period")
        data['RSI'] = calculate_rsi(data, period=rsi_period)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom left")
        fig_rsi.update_layout(title="Relative Strength Index", xaxis_title="Date", yaxis_title="RSI")
        st.plotly_chart(fig_rsi)

        # MACD
        st.subheader("Moving Average Convergence Divergence (MACD)")
        macd, signal = calculate_macd(data)
        data['MACD'] = macd
        data['Signal'] = signal
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal'], mode='lines', name='Signal Line'))
        fig_macd.update_layout(title="MACD", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_macd)

        # Display Data
        st.write("Technical Indicator Data")
        st.dataframe(data[[f"SMA_{ma_period}", 'RSI', 'MACD', 'Signal']].tail())
