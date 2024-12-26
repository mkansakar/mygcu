import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def compute_trend_slope():
    """
    Compute the slope of trend lines using linear regression on rolling windows.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Slope of Trend Lines")

    # Select column for slope calculation
    data = st.session_state['data']
    column = st.selectbox("Select a column for trend slope calculation", data.columns, index=data.columns.get_loc("Close"))

    # Rolling window size input
    st.subheader("Set Rolling Window Size")
    window_size = st.slider("Select Rolling Window Size (number of days)", min_value=5, max_value=50, value=20)

    # Function to calculate slope for each rolling window
    def rolling_slope(series, window):
        slopes = np.full(series.shape, np.nan)  # Initialize with NaN values
        for i in range(window - 1, len(series)):
            y = series[i - window + 1:i + 1].values
            x = np.arange(len(y)).reshape(-1, 1)  # x is 0, 1, 2, ..., window-1
            model = LinearRegression()
            model.fit(x, y)
            slopes[i] = model.coef_[0]  # Slope of the fitted line
        return slopes

    # Calculate the rolling slopes
    slopes = rolling_slope(data[column], window_size)

    # Add slope to a DataFrame for visualization
    slope_df = pd.DataFrame({
        "Date": data.index,
        column: data[column],
        "Slope": slopes
    }).dropna()  # Remove rows with NaN slopes

    # Display original data with slope
    st.subheader("Slope of Trend Lines Over Time")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the original data
    ax1.plot(data.index, data[column], label="Original Data", color="blue", alpha=0.7)
    ax1.set_ylabel(f"{column} Price", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot the slope
    ax2 = ax1.twinx()
    ax2.plot(slope_df["Date"], slope_df["Slope"], label="Slope", color="red", alpha=0.7)
    ax2.set_ylabel("Slope", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()
    st.pyplot(fig)

    # Store slopes in session state
    st.session_state['trend_slopes'] = slope_df
    with st.expander("What is Trend Slope?"):
            st.write("""
                A moving average in stock prices is a statistical measure used to analyze and smooth out price data by creating a continuously updated average price over a specific period.\n
                Bullish Signal: When the short-term average (5-day) crosses above the medium or long-term average (15-day or 30-day), it indicates rising momentum and a potential buy opportunity.\n
                Bearish Signal: When the short-term average crosses below the longer averages, it signals weakening momentum and a potential sell opportunity.
            """)  
    # Display slope DataFrame
    #st.subheader("Trend Slope Data")
    #st.write(slope_df)
