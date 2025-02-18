#stochastic_analysis.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def calculate_stochastic(data, k_period=14, d_period=3, column='Close'):
    """
    Calculate the Stochastic Oscillator.
    Args:
        data (DataFrame): The stock price data.
        k_period (int): The lookback period for %K.
        d_period (int): The moving average period for %D.
        column (str): The column used to calculate the Stochastic Oscillator.

    Returns:
        DataFrame: A DataFrame containing %K and %D values.
    """
    high_max = data['High'].rolling(window=k_period).max()
    low_min = data['Low'].rolling(window=k_period).min()

    # %K calculation
    k = ((data[column] - low_min) / (high_max - low_min)) * 100

    # %D calculation (SMA of %K)
    d = k.rolling(window=d_period).mean()

    return pd.DataFrame({'%K': k, '%D': d}, index=data.index)

def stochastic_analysis():
    """
    Perform stochastic analysis and visualize %K and %D indicators.
    """

    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Stochastic Analysis")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    #data = st.session_state['data']
    data = st.session_state['data'].tail(252)

    # Inputs for Stochastic Oscillator
    k_period = st.slider("Select %K Period", min_value=5, max_value=20, value=14, step=1)
    d_period = st.slider("Select %D Period (SMA of %K)", min_value=3, max_value=10, value=3, step=1)

    if 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
        st.error("The dataset must include 'High', 'Low', and 'Close' columns.")
        return

    # Calculate Stochastic Oscillator
    stochastic_df = calculate_stochastic(data, k_period=k_period, d_period=d_period)

    #st.subheader("Stochastic Oscillator Values")
    #st.write(stochastic_df.tail())

    # Plot %K and %D
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stochastic_df.index, y=stochastic_df['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=stochastic_df.index, y=stochastic_df['%D'], mode='lines', name='%D', line=dict(dash='dot')))
    fig.update_layout(
        title="Stochastic Oscillator",
        xaxis_title="Date",
        yaxis_title="Stochastic Value",
        template="plotly_white"
    )
    st.plotly_chart(fig)
    with st.expander("What is Stochastic Analysis?"):
        st.write("""
            The Stochastic Oscillator is a momentum indicator commonly used in technical analysis to assess the closing price of a security relative to its price range over a specific period.\n 
            A bullish signal occurs when %K crosses above %D in the oversold region.\n
            A bearish signal occurs when %K crosses below %D in the overbought region.\n
            Overbought (Above 80): Indicates that the price is trading near the upper end of its recent range and may be due for a correction or reversal.\n
            Oversold (Below 20): Suggests the price is trading near the lower end of its recent range and may be due for a rebound. \n    
            If the price makes a new high but the oscillator does not, it indicates weakening momentum (bearish divergence).\n
            If the price makes a new low but the oscillator does not, it indicates strengthening momentum (bullish divergence).

        """)