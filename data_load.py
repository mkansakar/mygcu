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

    # if (end_date - start_date).days > 365:
    #     st.warning("The date range cannot exceed 365 days. Please adjust the dates.")
    #     return
    
    if st.button("Load Data"):
        try:
            stock = yf.Ticker(stock_symbol)
            full_name = stock.info.get("longName", "N/A")
            data = stock.history(start=start_date, end=end_date)
            
            if not data.empty:
                st.subheader(f"{stock_symbol} Closing Price")
                fig = px.line(data, x=data.index, y="Close")
                fig.update_xaxes(title="Date")
                fig.update_yaxes(title="Price")
                st.plotly_chart(fig)

                st.session_state['data'] = data.iloc[:, :-2]
                st.session_state['symbol'] = full_name
                st.session_state['stock_symbol'] = stock_symbol
                
                # Load fundamental data automatically
                display_fundamentals(stock_symbol)

            else:
                st.warning(f"No data found for {stock_symbol} in the specified date range.")
        except Exception as e:
            st.error(f"Error downloading data: {e}")

    st.subheader("DISCLAIMER")
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

def display_fundamentals(symbol):
    """
    Display fundamental stock data.
    """
    st.subheader("Company Fundamentals")
    stock = yf.Ticker(symbol)
    
    info = stock.info
    info_table = {key: info[key] for key in ('shortName', 'sector', 'industry', 'country', 'website') if key in info}
    company_info = pd.DataFrame(info_table.items(), columns=["Attribute", "Value"])

    key_stats = {
        "EPS (TTM)": info.get("trailingEps"),
        "Market Cap": info.get("marketCap"),
        "PE Ratio (TTM)": info.get("trailingPE"),
        "Dividend Yield": info.get("dividendYield"),
        "52 Week High": info.get("fiftyTwoWeekHigh"),
        "52 Week Low": info.get("fiftyTwoWeekLow")
    }
    key_statistics = pd.DataFrame(key_stats.items(), columns=["Attribute", "Value"])

    col1, col2 = st.columns(2)
    with col1:
        st.table(company_info)
    with col2:
        st.table(key_statistics)

    with st.expander("What is Company Statistics?"):
        st.write("""
        EPS represents a company's profitability on a per-share basis. It shows how much profit is allocated to each outstanding share of common stock. A higher EPS generally indicates greater profitability.\n

        Market Cap measures the total value of a company's outstanding shares in the stock market. Larger companies tend to be more stable, while smaller ones may offer higher growth potential but carry more risk..\

        The PE Ratio compares a company's stock price to its earnings per share, reflecting how much investors are willing to pay for each dollar of earnings.\n
        High PE Ratio: Indicates investors expect higher growth in the future, but it may also signal overvaluation.\n
        Low PE Ratio: Suggests undervaluation or slow growth expectations.\n

        Dividend Yield represents the annual dividend payout as a percentage of the stock's current price.\n
        High Dividend Yield: Often attractive to income-focused investors but could indicate a declining stock price or unsustainable dividends.\n
        Low Dividend Yield: Suggests a focus on growth rather than income. A useful metric for investors seeking regular income.
        """)

  