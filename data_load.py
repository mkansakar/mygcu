#data_load.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

def load_data():
    st.title("Load Stock Price Data")

    end_date_default = datetime.today()
    start_date_default = end_date_default - timedelta(days=1825)

    col1, col2, col3 = st.columns(3)

    with col1:
        stock_symbol = st.text_input("Enter ticker (e.g., AAPL):", value="AAPL").upper()

    with col2:
        start_date = st.date_input("Start Date", value=start_date_default, max_value=end_date_default)

    with col3:
        end_date = st.date_input("End Date", value=end_date_default, max_value=end_date_default)

    # Validate date range
    date_range_valid = (end_date - start_date).days >= 365

    if start_date >= end_date:
        st.error("Start Date must be before End Date. Please adjust the dates.")
        date_range_valid = False
    elif not date_range_valid:
        st.warning("The date range must be at least 365 days. Please adjust the dates.")

    # "Load Data" button is always shown but only enabled when the date range is valid
    load_button = st.button("Load Data", disabled=not date_range_valid)

    if load_button and date_range_valid:
        try:
            st.session_state['session_data'] = None
            st.session_state['filtered_features'] = None
            stock = yf.Ticker(stock_symbol)
            full_name = stock.info.get("longName", "N/A")
            data = stock.history(start=start_date, end=end_date)
            data.index = pd.to_datetime(data.index).date



            gold = yf.Ticker("GC=F").history(start=start_date, end=end_date)
            gold = gold[['Close']].rename(columns={'Close': 'Gold_Close'})
            gold.index = pd.to_datetime(gold.index).date
            
            # Fetch GBP/USD Exchange Rate (GBPUSD=X)
            pound = yf.Ticker("GBPUSD=X").history(start=start_date, end=end_date)
            pound = pound[['Close']].rename(columns={'Close': 'GBPUSD'})
            pound.index = pd.to_datetime(pound.index).date
            #st.write(pound.tail(2)) 

            
            data = data.join(gold, how='left')
            data = data.join(pound, how='left')

            if not data.empty:
                st.subheader(f"{stock_symbol} Closing Price")
                fig = px.line(data, x=data.index, y="Close")
                fig.update_xaxes(title="Date")
                fig.update_yaxes(title="Price")
                st.plotly_chart(fig)

                st.write(data.tail(2))

                #st.session_state['data'] = data.iloc[:, :-2]
                st.session_state['session_data'] = data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
                #data = data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
                st.session_state['symbol'] = full_name
                st.session_state['stock_symbol'] = stock_symbol
                
                # Load fundamental data automatically
                display_fundamentals(stock_symbol)

            else:
                st.warning(f"No data found for {stock_symbol} in the specified date range.")
                #st.session_state['data'] = None
        except Exception as e:
            st.error(f"Error downloading data: {e}")

    st.subheader("DISCLAIMER")
    with st.expander("**DISCLAIMER: Please read before proceeding.**", expanded=True):
        st.write("""        
        The Stock Price Forecast Data Product is intended for informational and educational purposes only. It provides analytical tools and forecasts based on historical stock data and mathematical models. The following points should be carefully considered by all users:

        1. Not Financial Advice: The forecasts, analyses, and insights provided by this product do not constitute financial, investment, or trading advice. Always consult a certified financial advisor before making any investment decisions.
        2. Market Risks: Stock markets are inherently volatile and unpredictable. Past performance is not indicative of future results. Users should be aware of the risks associated with trading and investing.
        3. Accuracy and Limitations: While we strive to provide accurate and reliable forecasts, no guarantee is made regarding the accuracy, completeness, or timeliness of the information presented. The models used are subject to limitations and assumptions, which may not account for all market variables or events.
        4. User Responsibility: The use of this product is at the user's own risk. The creators and developers of this product are not liable for any financial losses or damages arising from the use of the information provided.
        5. No Warranty: This product is provided "as is," without any warranties or guarantees of any kind, either expressed or implied.

        By using this product, you acknowledge and agree to the terms of this disclaimer. Always perform your due diligence and exercise caution when making financial decisions.
        """)


def display_fundamentals(symbol):
    """
    Display fundamental stock data with exception handling.
    """
    try:
        st.subheader("Company Fundamentals")

        # Fetch stock data from Yahoo Finance
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
        except Exception as e:
            st.error(f"Error fetching stock data for {symbol}: {e}")
            return

        # Ensure the info dictionary contains data
        if not info or "shortName" not in info:
            st.error(f"No fundamental data available for {symbol}.")
            return

        # Extract company info safely
        try:
            info_table = {key: info[key] for key in ('shortName', 'sector', 'industry', 'country', 'website') if key in info}
            company_info = pd.DataFrame(info_table.items(), columns=["Attribute", "Value"])
        except Exception as e:
            st.error(f"Error extracting company info: {e}")
            company_info = pd.DataFrame(columns=["Attribute", "Value"])  # Empty DataFrame fallback

        # Extract key statistics safely
        try:
            key_stats = {
                "EPS (TTM)": info.get("trailingEps", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "PE Ratio (TTM)": info.get("trailingPE", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
                "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
            }
            key_statistics = pd.DataFrame(key_stats.items(), columns=["Attribute", "Value"])
        except Exception as e:
            st.error(f"Error extracting key statistics: {e}")
            key_statistics = pd.DataFrame(columns=["Attribute", "Value"])  # Empty DataFrame fallback

        # Display company info and key statistics
        col1, col2 = st.columns(2)
        with col1:
            if not company_info.empty:
                st.table(company_info)
            else:
                st.warning("Company info is missing.")

        with col2:
            if not key_statistics.empty:
                st.table(key_statistics)
            else:
                st.warning("Key statistics are missing.")

        # Expandable section for explanation
        with st.expander("What is Company Statistics?"):
            st.write("""
            **EPS (Earnings Per Share)** represents a company's profitability on a per-share basis.
            It shows how much profit is allocated to each outstanding share of common stock. 
            A higher EPS generally indicates greater profitability.\n

            **Market Cap** measures the total value of a company's outstanding shares in the stock market. 
            Larger companies tend to be more stable, while smaller ones may offer higher growth potential but carry more risk.\n

            **PE Ratio (Price-to-Earnings Ratio)** compares a company's stock price to its earnings per share, 
            reflecting how much investors are willing to pay for each dollar of earnings.\n
            - High PE Ratio: Indicates investors expect higher growth in the future, but it may also signal overvaluation.\n
            - Low PE Ratio: Suggests undervaluation or slow growth expectations.\n

            **Dividend Yield** represents the annual dividend payout as a percentage of the stock's current price.\n
            - High Dividend Yield: Often attractive to income-focused investors but could indicate a declining stock price or unsustainable dividends.\n
            - Low Dividend Yield: Suggests a focus on growth rather than income. A useful metric for investors seeking regular income.
            """)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

  