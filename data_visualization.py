import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def visualize_data():
    """
    Visualize stock data as a candlestick chart.
    """
    if 'data' not in st.session_state:
        st.error("Please load the data first from the sidebar on the left.")
        return
    
    data = st.session_state['data']
    st.write(data.tail())
    
    st.title("Candlestick Chart")
    #st.markdown(f"Stock: {st.session_state['symbol']}")
    # Ensure the required columns are present for the candlestick chart
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']
                )
            ]
        )
        fig.update_layout(
            title="Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("The dataset does not contain required columns for a Candlestick Chart.")

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
