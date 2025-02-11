#decompose.py
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def decompose_time_series():
    """
    Decompose the time series into trend, seasonal, and residual components.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Time Series Decomposition")

    # Select column for decomposition
    data = st.session_state['data']
    column = st.selectbox("Select a column for decomposition", data.columns, index=data.columns.get_loc("Close"))

    # Ensure the data has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        st.error("Time series data must have a datetime index.")
        return

    # Select decomposition model
    st.subheader("Decomposition Options")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    model = st.radio("Select Decomposition Model", ["Additive", "Multiplicative"], index=0)

    # Perform decomposition
    st.write("Performing decomposition...")
    try:
        decomposition = seasonal_decompose(data[column], model=model, period=30)  # Assume monthly seasonality for daily data

        # Plot the decomposed components
        st.subheader("Decomposition Components")
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        decomposition.observed.plot(ax=axes[0], title="Observed")
        decomposition.trend.plot(ax=axes[1], title="Trend")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
        decomposition.resid.plot(ax=axes[3], title="Residual")

        for ax in axes:
            ax.grid()

        st.pyplot(fig)

        # Save decomposed components in session state
        st.session_state['decomposition'] = {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid
        }

    except ValueError as e:
        st.error(f"Decomposition failed: {str(e)}")
    with st.expander("What is Time series Decomposition?"):
        st.write("""
            Time series decomposition in stock price forecasting is a method of breaking down a time series into its underlying components to better understand its structure and behavior.\n
            Trend represents the overall direction of stock prices over a longer period, such as an upward or downward trajectory. It helps investors identify long-term growth or decline, ignoring short-term fluctuations.\n
            Seasonality captures recurring patterns or cycles within specific periods, such as weekly, monthly, or yearly behaviors. For example, retail stocks might show higher prices during the holiday season due to increased sales activity.\n
            Residuals or noise represent random, unpredictable fluctuations in stock prices after accounting for the trend and seasonality. These variations are often caused by unexpected market events or speculative behavior and highlight the inherent unpredictability of stock prices.\n          
        """)  