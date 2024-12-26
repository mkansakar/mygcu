import streamlit as st
import pandas as pd

def calculate_price_returns():
    """
    Compute weekly and monthly percentage changes in stock prices.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
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
        st.write("Weekly percentage changes:")
        #st.write(weekly_returns.tail())
        st.line_chart(weekly_returns, use_container_width=True)

        # Store computed returns in session state
        st.session_state['weekly_returns'] = weekly_returns

    elif timeframe == "Monthly":
        # Calculate monthly percentage changes
        st.subheader("Monthly Percentage Changes")
        monthly_returns = data[column].pct_change(periods=21) * 100  # Assuming 21 trading days per month
        monthly_returns.dropna(inplace=True)

        # Display results
        st.write("Monthly percentage changes:")
        #st.write(monthly_returns.tail())
        st.line_chart(monthly_returns, use_container_width=True)

        # Store computed returns in session state
        st.session_state['monthly_returns'] = monthly_returns
    with st.expander("What is Price Return?"):
        st.write("""
            Price return analysis evaluates the percentage change in a stock’s price over a specific period, offering insights into its performance and potential trends.\n
            Positive Return: A positive percentage return indicates that the stock’s price has increased during the analyzed timeframe.\n
                 A 5% weekly return means the stock's price grew by 5% over the past week.
            Negative Return: A negative percentage return suggests that the stock’s price has decreased. \n
                 A -3% daily return indicates a 3% drop in the stock's price from the previous day.
        """)  
