#moving.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go 

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
    Display combined moving averages, RSI, and MACD plots without modifying the original dataset.
    """
    

    if 'data' not in st.session_state:
        st.error("Please load the data first from the sidebar on the left.")
        return
    st.title("Moving Averages and Indicators")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    data = st.session_state['data']

    # Moving Averages
    st.header("Moving Averages")
    windows = st.multiselect("Select Moving Average Periods", [5, 15, 30], default=[5, 15, 30])
    if windows:
        moving_avg = calculate_moving_averages(data, windows=windows)
        fig = go.Figure()
        
        # Add original Close prices to the graph
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Close',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving averages to the graph
        for period, avg in moving_avg.items():
            fig.add_trace(go.Scatter(
                x=data.index, y=avg, mode='lines', name=f"{period}",
                line=dict(width=2)
            ))

        # Configure layout
        fig.update_layout(
            title="Combined Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_white"
        )
        st.plotly_chart(fig)
    with st.expander("What is Moving Averages?"):
            st.write("""
                A moving average in stock prices is a statistical measure used to analyze and smooth out price data by creating a continuously updated average price over a specific period.\n
                Bullish Signal: When the short-term average (5-day) crosses above the medium or long-term average (15-day or 30-day), it indicates rising momentum and a potential buy opportunity.\n
                Bearish Signal: When the short-term average crosses below the longer averages, it signals weakening momentum and a potential sell opportunity.
            """)  
    # RSI
    st.header("Relative Strength Index (RSI)")
    rsi_window = st.slider("Select RSI Window", 10, 30, 14)
    rsi = calculate_rsi(data, window=rsi_window)
    st.line_chart(pd.DataFrame({"RSI": rsi}, index=data.index))
    with st.expander("What is RSI?"):
            st.write("""
                The RSI is a momentum oscillator that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions in a stock or other financial instrument.\n
                If a stock has an RSI of 80, it is overbought, and traders may expect a price correction.\n
                If a stock has an RSI of 25, it is oversold, and traders might anticipate a price recovery.\n
                If RSI moves from below 30 to above 30, it can signal the end of a downtrend and the beginning of an upward move.
            """)  
    
    # MACD
    st.header("MACD")
    short_window = st.slider("Short EMA Window", 5, 20, 12)
    long_window = st.slider("Long EMA Window", 20, 50, 26)
    signal_window = st.slider("Signal Line Window", 5, 20, 9)
    macd, signal_line = calculate_macd(data, short_window=short_window, long_window=long_window, signal_window=signal_window)
    macd_df = pd.DataFrame({"MACD": macd, "Signal Line": signal_line}, index=data.index)
    st.line_chart(macd_df)

    #st.info("Note: These plots are based on calculated values and do not modify the original dataset.")
 
    with st.expander("What is MACD?"):
            st.write("""
                The MACD is a trend-following momentum indicator used in technical analysis to identify changes in the strength, direction, momentum, and duration of a trend in a stock’s price.\n
                Buy Signal: When the MACD line crosses above the Signal line, it indicates bullish momentum. This is interpreted as a signal to buy or go long on the stock.\n
                Sell Signal: When the MACD line crosses below the Signal line, it suggests bearish momentum. This is seen as a signal to sell or go short.
            """)    