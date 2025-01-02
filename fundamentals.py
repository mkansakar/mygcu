import streamlit as st
import yfinance as yf
import pandas as pd

# Define the stock symbol (e.g., AAPL for Apple Inc.)
def fundamentals():
    
    if 'symbol' not in st.session_state:
        st.error("Please load the data first from the sidebar on the left.")
        return
    st.title("Stock Fundamentals")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    # User Input for Stock Symbol
    symbol = st.session_state['stock_symbol']

    # Load Data if Symbol Provided
    if symbol:
        try:
            # Fetch Stock Data
            stock = yf.Ticker(symbol)
            
            # Display Stock Info
            #st.header(f"Company Information")
            info = stock.info
            info_table = {key: info[key] for key in ('shortName', 'sector', 'industry', 'country', 'website') if key in info}
            company_info = pd.DataFrame(info_table.items(), columns=["Attribute", "Value"])
            #st.table(info_table.items())

            # Display Key Statistics
            #st.header("Key Statistics")
            key_stats = {
                "EPS (TTM)": info.get("trailingEps"),
                "Market Cap": info.get("marketCap"),
                "PE Ratio (TTM)": info.get("trailingPE"),
                "Dividend Yield": info.get("dividendYield"),
                "52 Week High": info.get("fiftyTwoWeekHigh"),
                "52 Week Low": info.get("fiftyTwoWeekLow")
            }
            key_statistics = pd.DataFrame(key_stats.items(), columns=["Attribute", "Value"])
            #st.table(key_stats.items())

            # Display tables side by side
            col1, col2 = st.columns(2)
            with col1:
                #st.subheader("Company Information")
                st.table(company_info)
            with col2:
                #st.subheader("Key Statistics")
                st.table(key_statistics)

            # Balance Sheet
            # st.header("Balance Sheet")
            # balancesheet = stock.balance_sheet
            # if not balancesheet.empty:
            #     st.write(balancesheet)
            # else:
            #     st.write("Balance sheet data not available.")

            # Income Statement
            # st.header("Income Statement")
            # income_statement = stock.financials
            # if not income_statement.empty:
            #     st.write(income_statement)
            # else:
            #     st.write("Income statement data not available.")

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

            # Cash Flow Statement
            st.header("Cash Flow Statement")
            cashflow = stock.cashflow
            if not cashflow.empty:
                st.write(cashflow)
            else:
                st.write("Cash flow data not available.")

            # Earnings History
            # st.header("Earnings History")
            # earnings = stock.earnings
            # if not earnings.empty:
            #     st.write(earnings)
            # else:
            #     st.write("Earnings data not available.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.info("Please enter a stock symbol to view its fundamentals.")

    
    with st.expander("What is Company Fundamental?"):
        st.write("""
            Company fundamentals refer to the core quantitative and qualitative information about a business that is used to evaluate its overall health, financial performance, and intrinsic value. These fundamentals are essential for making informed investment decisions and are typically divided into financial metrics and business-specific factors.\n
            info: Provides detailed company information, such as market capitalization, PE ratio, dividend yield, etc..\n
            balance_sheet: Contains the company's assets, liabilities, and equity.\n
            financials: Displays the income statement including revenue, operating expenses, and net income.\n
            cashflow: Shows cash flow activities including operating, investing, and financing activities.\n
            major_holders: Lists major shareholders.\n
            institutional_holders: Provides institutional ownership details.\n
            recommendations: Analyst recommendations with dates.\n
        """)