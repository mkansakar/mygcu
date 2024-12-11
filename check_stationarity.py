import streamlit as st
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def check_stationarity():
    """
    Checks the stationarity of the dataset using the Augmented Dickey-Fuller (ADF) test.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("No data found. Please load data first.")
        return

    st.title("Check Stationarity")
    data = st.session_state['data']

    # Dropdown to select a column to check for stationarity
    column = st.selectbox("Select a column for stationarity check", data.columns, index=data.columns.get_loc("Close"))
    
    # Perform the Augmented Dickey-Fuller test
    st.write(f"Performing stationarity check on column: {column}")
    result = adfuller(data[column].dropna())
    st.write("Augmented Dickey-Fuller Test Results:")
    st.write({
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Number of Lags Used": result[2],
        "Number of Observations Used": result[3]
    })

    # Interpretation
    if result[1] < 0.05:
        st.success(f"The data in column '{column}' is stationary (p-value = {result[1]}).")
    else:
        st.warning(f"The data in column '{column}' is not stationary (p-value = {result[1]}). Consider differencing or other transformations.")

    # Provide recommendations if non-stationary
    if result[1] >= 0.05:
        st.subheader("Recommended Transformations")
        st.write("""
        - Apply differencing to the series.
        - Log transformation.
        - Detrend the data.
        """)

