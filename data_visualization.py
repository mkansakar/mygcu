#data_visualization.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def identify_candlestick_pattern(data):
    """
    Determines the next day's price trend based on the last candlestick pattern.
    """
    last_candle = data.iloc[-1]

    # Engulfing Pattern (Bullish)
    if last_candle['Close'] > last_candle['Open'] and data.iloc[-2]['Close'] < data.iloc[-2]['Open']:
        if last_candle['Close'] > data.iloc[-2]['Open'] and last_candle['Open'] < data.iloc[-2]['Close']:
            return "Bullish Engulfing - Uptrend Expected"

    # Engulfing Pattern (Bearish)
    if last_candle['Close'] < last_candle['Open'] and data.iloc[-2]['Close'] > data.iloc[-2]['Open']:
        if last_candle['Close'] < data.iloc[-2]['Open'] and last_candle['Open'] > data.iloc[-2]['Close']:
            return "Bearish Engulfing - Downtrend Expected"

    # Hammer (Bullish)
    if (last_candle['Close'] > last_candle['Open']) and ((last_candle['Low'] < last_candle['Open'] * 0.98) and (last_candle['High'] - last_candle['Close']) < (last_candle['Close'] - last_candle['Open'])):
        return "Hammer - Potential Uptrend"

    # Shooting Star (Bearish)
    if (last_candle['Close'] < last_candle['Open']) and ((last_candle['High'] > last_candle['Open'] * 1.02) and (last_candle['Close'] - last_candle['Low']) < (last_candle['Open'] - last_candle['Close'])):
        return "Shooting Star - Potential Downtrend"

    return "No Clear Pattern - Market Indecision"

def predict_next_day_trend():
    """
    Display the next day's expected price trend based on candlestick analysis.
    """
    if 'data' not in st.session_state:
        st.error("Please load the data first.")
        return

    data = st.session_state['data']
    
    # Ensure there are at least two candles
    if len(data) < 2:
        st.error("Not enough data to detect candlestick patterns.")
        return

    pattern = identify_candlestick_pattern(data)
    
    #st.subheader("Next Day Price Prediction (Candlestick Analysis)")
    st.write("__Candlestick Analysis__:")
    st.markdown(f"**{pattern}**")


def visualize_data():
    """
    Visualize stock data as a candlestick chart.
    """
    if 'data' not in st.session_state:
        st.error("Please load the data first from the sidebar on the left.")
        return
    
    data = st.session_state['data']
    
    # Ensure the data has at least 30 days of records
    if len(data) < 30:
        st.error("Not enough data to display a candlestick chart for the last 30 days.")
        return
    
    # Slice the last 30 days of data
    last_30_days = data.tail(252)    
    
    st.title("Candlestick Chart")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    # Ensure the required columns are present for the candlestick chart
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        #st.title("Candlestick Chart")

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=last_30_days.index,
                    open=last_30_days['Open'],
                    high=last_30_days['High'],
                    low=last_30_days['Low'],
                    close=last_30_days['Close']
                )
            ]
        )

        fig.update_layout(
            #title="Candlestick Chart (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )

        st.plotly_chart(fig)

        predict_next_day_trend()
    else:
        st.error("The dataset does not contain the required columns for a candlestick chart.")


    # Add a link for basic information about candlestick charts
    with st.expander("What is a Candlestick Chart?"):
        st.write("""
            A candlestick chart is a style of financial chart used to describe price movements of a security, derivative, or currency.\n 
            Bullish Patterns:\n
            Hammer: A small body at the top with a long lower wick, signaling a potential reversal to an upward trend.\n
            Engulfing Pattern: A green candlestick completely engulfs the previous red candlestick, indicating bullish momentum.\n
            Bearish Patterns:\n
            Shooting Star: A small body at the bottom with a long upper wick, signaling a potential reversal to a downward trend.\n
            Bearish Engulfing: A red candlestick completely engulfs the previous green candlestick, signaling bearish momentum.
        """)
