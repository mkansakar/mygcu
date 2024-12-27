import streamlit as st
import numpy as np
import pandas as pd

def detrend_series(series):
    """
    Remove trends from a series using linear detrending.
    """
    x = np.arange(len(series))
    y = series.values
    coeffs = np.polyfit(x, y, 1)  # Fit a linear trend (degree=1)
    trend = coeffs[0] * x + coeffs[1]
    detrended = y - trend
    return pd.Series(detrended, index=series.index)

def apply_transformations():
    """
    Apply data transformations such as differencing, log transformation, and detrending.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Data Transformations")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    # Select column for transformation
    data = st.session_state['data']
    column = st.selectbox("Select a column for transformation", data.columns, index=data.columns.get_loc("Close"))

    # Display the original data
    st.subheader("Original Data")
    st.line_chart(data[column])

    # Transformation options
    st.subheader("Available Transformations")

    # Apply differencing
    if st.checkbox("Apply Differencing"):
        differenced_data = data[column].diff().dropna()
        st.subheader("Differenced Data")
        st.line_chart(differenced_data)
        st.session_state['differenced_data'] = differenced_data  # Store the transformed data in session

    # Apply log transformation
    if st.checkbox("Apply Log Transformation"):
        if (data[column] <= 0).any():
            st.warning(f"Log transformation cannot be applied to column '{column}' because it contains non-positive values.")
        else:
            log_transformed_data = np.log(data[column])
            st.subheader("Log Transformed Data")
            st.line_chart(log_transformed_data)
            st.session_state['log_transformed_data'] = log_transformed_data  # Store the transformed data in session

    # Apply detrending
    if st.checkbox("Apply Detrending"):
        detrended_data = detrend_series(data[column])
        st.subheader("Detrended Data")
        st.line_chart(detrended_data)
        st.session_state['detrended_data'] = detrended_data  # Store the transformed data in session
