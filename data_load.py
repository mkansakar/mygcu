import streamlit as st
import yfinance as yf
import pandas as pd

def load_data():
    """
    Load stock data for the selected symbol and date range.
    """
    st.title("Load Stock Price Data")

    # Input fields for stock symbol and date range
    #stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):").upper()
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-07-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    # Restrict the date range to 365 days only
    if (end_date - start_date).days > 365:
        st.warning("The date range cannot exceed 365 days. Please adjust the dates.")
        return

    # Fetch the data when the button is clicked
    if st.button("Load Data"):
        try:
            data = yf.download(stock_symbol, start=start_date, end=end_date)

            if data.empty:
                st.error(f"No data found for {stock_symbol} between {start_date} and {end_date}. Please check the symbol and date range.")
            else:
                st.session_state['data'] = data
                st.success(f"Data loaded successfully for {stock_symbol}!")
                st.dataframe(data.head())  # Display the first few rows of data
        except Exception as e:
            st.error(f"An error occurred while fetching the data: {str(e)}")
