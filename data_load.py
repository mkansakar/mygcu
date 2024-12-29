import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

def load_data():
    """
    Load stock data from Yahoo Finance.
    """
    st.title("Load Stock Price Data")
    #st.markdown("<h2 style='text-align: center; font-size: 20px;'>Load Stock Price Data</h2>", unsafe_allow_html=True)
    end_date_default = datetime.today()
    start_date_default = end_date_default - timedelta(days=365)

    # Use st.columns to place inputs side by side
    col1, col2, col3 = st.columns(3)

    # Input for stock symbol
    with col1:
        stock_symbol = st.text_input("Enter ticker (e.g., AAPL):", value="AAPL").upper()

    # Input for start date
    with col2:
        start_date = st.date_input("Start Date", value=start_date_default, max_value=end_date_default)

    # Input for end date below the same row
    with col3:
        end_date = st.date_input("End Date", value=end_date_default, max_value=end_date_default)

    # Restrict the date range to 365 days only
    if (end_date - start_date).days > 365:
        st.warning("The date range cannot exceed 365 days. Please adjust the dates.")
        return
    
    # Fetch the data when the button is clicked
    if st.button("Load Data"):
        try:
            stock = yf.Ticker(stock_symbol)
            full_name = stock.info.get("longName", "N/A")
            data = stock.history(start=start_date, end=end_date)
            if not data.empty:
                #st.success(f"Data successfully loaded for {stock_symbol} from {start_date} to {end_date}.")
                #st.subheader(f"{stock_symbol} Closing Price")
                fig = px.line(data, x=data.index, y="Close", title=f"Closing Price Over Time - {full_name}")
                fig.update_xaxes(title="Date")
                fig.update_yaxes(title="Price")
                st.plotly_chart(fig)                
                #st.write(data.tail())  # Display the first few rows
                st.session_state['data'] = data  # Store data in session state
                st.session_state['symbol'] = full_name
            else:
                st.warning(f"No data found for {stock_symbol} in the specified date range.")
        except Exception as e:
            st.error(f"Error downloading data: {e}")



    with st.expander("**DISCLAIMER: Please read before proceeding.**"):
        st.write("""        
        The Stock Price Forecast Data Product is intended for informational and educational purposes only. It provides analytical tools and forecasts based on historical stock data and mathematical models. The following points should be carefully considered by all users:

        1. Not Financial Advice: The forecasts, analyses, and insights provided by this product do not constitute financial, investment, or trading advice. Always consult a certified financial advisor before making any investment decisions.
        2. Market Risks: Stock markets are inherently volatile and unpredictable. Past performance is not indicative of future results. Users should be aware of the risks associated with trading and investing.
        3. Accuracy and Limitations: While we strive to provide accurate and reliable forecasts, no guarantee is made regarding the accuracy, completeness, or timeliness of the information presented. The models used are subject to limitations and assumptions, which may not account for all market variables or events.
        4. User Responsibility: The use of this product is at the user's own risk. The creators and developers of this product are not liable for any financial losses or damages arising from the use of the information provided.
        5. No Warranty: This product is provided "as is," without any warranties or guarantees of any kind, either expressed or implied.

        By using this product, you acknowledge and agree to the terms of this disclaimer. Always perform your due diligence and exercise caution when making financial decisions.
        """)  