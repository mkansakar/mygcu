import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_data():
    """
    Load stock data for the selected symbol and date range.
    """
    st.title("Load Stock Price Data")

    # Input fields for stock symbol and date range
    #stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):").upper()
    # Set default dates: end_date is current date, start_date is end_date - 364 days
    end_date_default = datetime.today()
    start_date_default = end_date_default - timedelta(days=364)

    # Input fields for stock symbol and date range
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=start_date_default, max_value=end_date_default)
    end_date = st.date_input("End Date", value=end_date_default, max_value=end_date_default)

    # Restrict the date range to 365 days only
    if (end_date - start_date).days > 365:
        st.warning("The date range cannot exceed 365 days. Please adjust the dates.")
        return

    # Fetch the data when the button is clicked
    if st.button("Load Data"):
        try:
            stock = yf.Ticker(stock_symbol)
            #data = yf.download(stock_symbol, start=start_date, end=end_date)
            
            data = stock.history(start=start_date, end=end_date)    
            if data.empty:
                st.error(f"No data found for {stock_symbol} between {start_date} and {end_date}. Please check the symbol and date range.")
            else:
                st.session_state['data'] = data
                st.success(f"Data loaded successfully for {stock_symbol}!")
                st.dataframe(data.head())  # Display the first few rows of data
        except Exception as e:
            st.error(f"An error occurred while fetching the data: {str(e)}")
    with st.expander("Loading Stock data:"):
        st.write("""
            Choose ticker name of your choice. For example APPL for Apple or NVDA for NVidia or MSFT for Microsoft.\n
            Choose start date and end date; make sure max days cannot exceed 365 days.
        """)  