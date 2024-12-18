import streamlit as st
import pandas as pd

def resample_data():
    """
    Resample the stock price data to different time intervals.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Data Resampling")

    # Load the original data
    data = st.session_state['data']

    # Ensure the data has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        st.error("Time series data must have a datetime index.")
        return

    # Select resampling frequency
    st.subheader("Resampling Options")
    frequency = st.selectbox("Select Resampling Frequency", ["Weekly", "Monthly", "Quarterly"], index=0)

    # Resample data based on selected frequency
    freq_mapping = {
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q"
    }

    resampled_data = data.resample(freq_mapping[frequency]).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # Display resampled data
    st.subheader(f"Resampled Data ({frequency})")
    st.dataframe(resampled_data.head())

    # Save resampled data in session state for further use
    st.session_state[f'resampled_data_{frequency.lower()}'] = resampled_data

    # Visualization
    st.subheader(f"Resampled Data Plot ({frequency})")
    st.line_chart(resampled_data["Close"], use_container_width=True)

    st.success(f"Data successfully resampled to {frequency} frequency.")
