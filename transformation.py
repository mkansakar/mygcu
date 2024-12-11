import streamlit as st
import numpy as np
import pandas as pd

def apply_transformations():
    """
    Apply data transformations such as differencing and log transformation.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("No data found. Please load data first.")
        return

    st.title("Data Transformations")

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
