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

        # Display components as dataframes
        st.subheader("Decomposed Components as DataFrame")
        trend_df = decomposition.trend.dropna().to_frame(name="Trend")
        seasonal_df = decomposition.seasonal.dropna().to_frame(name="Seasonal")
        residual_df = decomposition.resid.dropna().to_frame(name="Residual")

        st.write("Trend Component")
        st.dataframe(trend_df)

        st.write("Seasonal Component")
        st.dataframe(seasonal_df)

        st.write("Residual Component")
        st.dataframe(residual_df)

    except ValueError as e:
        st.error(f"Decomposition failed: {str(e)}")
