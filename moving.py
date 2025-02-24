import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go 

def calculate_rsi(data, column='Close', window=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD and Signal Line.
    """
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# def predict_next_day_trend(rsi, macd, signal_line):
#     """
#     Predict next day's trend using RSI and MACD.
#     """
#     latest_rsi = rsi.iloc[-1]
#     latest_macd = macd.iloc[-1]
#     latest_signal = signal_line.iloc[-1]

#     if latest_rsi < 30 and latest_macd > latest_signal:
#         return "Uptrend (RSI Oversold & MACD Bullish)"
#     elif latest_rsi > 70 and latest_macd < latest_signal:
#         return "Downtrend (RSI Overbought & MACD Bearish)"
#     elif latest_macd > latest_signal:
#         return "Uptrend (MACD Bullish Crossover)"
#     elif latest_macd < latest_signal:
#         return "Downtrend (MACD Bearish Crossover)"
#     else:
#         return "Neutral"

def predict_rsi_trend(latest_rsi):
    """
    Predict trend based on RSI alone.
    """
    if latest_rsi < 30:
        return "Uptrend (RSI Oversold)"
    elif latest_rsi > 70:
        return "Downtrend (RSI Overbought)"
    else:
        return "Neutral (RSI in Normal Range)"

def predict_macd_trend(latest_macd, latest_signal):
    """
    Predict trend based on MACD crossover.
    """
    if latest_macd > latest_signal:
        return "Uptrend (MACD Bullish Crossover)"
    elif latest_macd < latest_signal:
        return "Downtrend (MACD Bearish Crossover)"
    else:
        return "Neutral (MACD No Crossover)"


def moving_indicators():
    """
    Display moving averages, RSI, and MACD plots with next-day trend prediction.
    """
    try:    
        if 'session_data' not in st.session_state or st.session_state['session_data'] is None:
                st.error("Please use Load Data button on left menu to load the data first.")
                return
        
        st.title("Moving Averages and Indicators")
        st.markdown(f"Stock: {st.session_state['symbol']}")
        
        data = st.session_state['session_data'].tail(252)

        # RSI Calculation
        st.header("Relative Strength Index (RSI)")
        rsi_window = st.slider("Select RSI Window", 10, 30, 14)
        rsi = calculate_rsi(data, window=rsi_window)
        st.line_chart(pd.DataFrame({"RSI": rsi}, index=data.index))
        next_day_rsi_trend = predict_rsi_trend(rsi.iloc[-1])
        st.write(f"**Predicted Next Day Trend: {next_day_rsi_trend}**")

        # MACD Calculation
        st.header("MACD")
        short_window = st.slider("Short EMA Window", 5, 20, 12)
        long_window = st.slider("Long EMA Window", 20, 50, 26)
        signal_window = st.slider("Signal Line Window", 5, 20, 9)
        macd, signal_line = calculate_macd(data, short_window=short_window, long_window=long_window, signal_window=signal_window)
        macd_df = pd.DataFrame({"MACD": macd, "Signal Line": signal_line}, index=data.index)
        st.line_chart(macd_df)

        # Predict Next Day Trend
        # next_day_trend = predict_macd_trend(macd, signal_line)
        next_day_macd_trend = predict_macd_trend(macd.iloc[-1], signal_line.iloc[-1])
        st.write(f"**Predicted Next Day Trend: {next_day_macd_trend}**")        
        # st.write(f"Predicted Next Day Trend: {next_day_trend}")

        # Additional Info
        with st.expander("How is the prediction made?"):
            st.write("""
            The trend prediction is based on RSI and MACD indicators:
            - **Uptrend**: RSI is below 30 (oversold) and MACD is crossing above the signal line (bullish).
            - **Downtrend**: RSI is above 70 (overbought) and MACD is crossing below the signal line (bearish).
            - **Neutral**: No strong confirmation from both indicators.
            """)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
