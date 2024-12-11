import streamlit as st
import pandas as pd

def calculate_price_returns():
    """
    Compute weekly and monthly percentage changes in stock prices.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("No data found. Please load data first.")
        return

    st.title("Price Returns Analysis")

    # Select column for price returns calculation
    data = st.session_state['data']
    column = st.selectbox("Select a column to compute percentage changes", data.columns, index=data.columns.get_loc("Close"))

    # Timeframe selection
    st.subheader("Select Timeframe for Price Returns")
    timeframe = st.radio("Choose a timeframe", options=["Weekly", "Monthly"], index=0)

    if timeframe == "Weekly":
        # Calculate weekly percentage changes
        st.subheader("Weekly Percentage Changes")
        weekly_returns = data[column].pct_change(periods=5) * 100  # Assuming 5 trading days per week
        weekly_returns.dropna(inplace=True)

        # Display results
        st.write("Weekly percentage changes (first few rows):")
        st.write(weekly_returns.head())
        st.line_chart(weekly_returns, use_container_width=True)

        # Store computed returns in session state
        st.session_state['weekly_returns'] = weekly_returns

    elif timeframe == "Monthly":
        # Calculate monthly percentage changes
        st.subheader("Monthly Percentage Changes")
        monthly_returns = data[column].pct_change(periods=21) * 100  # Assuming 21 trading days per month
        monthly_returns.dropna(inplace=True)

        # Display results
        st.write("Monthly percentage changes (first few rows):")
        st.write(monthly_returns.head())
        st.line_chart(monthly_returns, use_container_width=True)

        # Store computed returns in session state
        st.session_state['monthly_returns'] = monthly_returns
