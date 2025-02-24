#stochastic_analysis.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def calculate_stochastic(data, k_period=14, d_period=3, column='Close'):
    """
    Calculate the Stochastic Oscillator and predict next day's price movement.
    """
    high_max = data['High'].rolling(window=k_period).max()
    low_min = data['Low'].rolling(window=k_period).min()

    # %K calculation
    k = ((data[column] - low_min) / (high_max - low_min)) * 100

    # %D calculation (SMA of %K)
    d = k.rolling(window=d_period).mean()

    # Identify overbought & oversold levels
    data['%K'] = k
    data['%D'] = d

    # Identify potential buy and sell signals
    data['Stochastic_Signal'] = "Neutral"

    # Bullish signal: %K crosses above %D and both are in oversold region (< 20)
    data.loc[(data['%K'] > data['%D']) & (data['%K'] < 20), 'Stochastic_Signal'] = "Bullish - Possible Uptrend"

    # Bearish signal: %K crosses below %D and both are in overbought region (> 80)
    data.loc[(data['%K'] < data['%D']) & (data['%K'] > 80), 'Stochastic_Signal'] = "Bearish - Possible Downtrend"

    return data[['%K', '%D', 'Stochastic_Signal']]

def determine_next_trend(data):
    """
    Determine the next day's price movement based on the Stochastic Oscillator.
    """
    if len(data) < 2:
        return "Not enough data to determine the trend."

    last_signal = data['Stochastic_Signal'].iloc[-1]

    if "Bullish" in last_signal:
        return "Likely Uptrend"
    elif "Bearish" in last_signal:
        return "Likely Downtrend"
    else:
        return "No Clear Trend"

def stochastic_analysis():
    try:

        """
        Perform stochastic analysis, visualize %K and %D indicators, and predict next-day movement.
        """
    
        if 'session_data' not in st.session_state or st.session_state['session_data'] is None:
            st.error("Please use Load Data button on left menu to load the data first.")
            return

        st.title("Stochastic Analysis")
        st.markdown(f"Stock: {st.session_state['symbol']}")

        data = st.session_state['session_data'].tail(252)

        # Inputs for Stochastic Oscillator
        k_period = st.slider("Select %K Period", min_value=5, max_value=20, value=14, step=1)
        d_period = st.slider("Select %D Period (SMA of %K)", min_value=3, max_value=10, value=3, step=1)

        if 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
            st.error("The dataset must include 'High', 'Low', and 'Close' columns.")
            return

        # Calculate Stochastic Oscillator
        stochastic_df = calculate_stochastic(data, k_period=k_period, d_period=d_period)

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

        # Display Next-Day Price Movement Prediction
        #st.subheader("Next Day Price Movement Prediction Based on Stochastic Analysis")
        
        trend_prediction = determine_next_trend(stochastic_df)
        st.write(f"**Stochastic Analysis: {trend_prediction}**")   

        with st.expander("What is Stochastic Analysis?"):
            st.write("""
                The Stochastic Oscillator is a momentum indicator used to assess the closing price of a stock relative to its price range over a specific period.
                
                Key Insights:
                - Overbought (> 80): The stock is trading near the upper range and may face a correction.
                - Oversold (< 20): The stock is trading near the lower range and may see a rebound.
                - Bullish Signal: When %K crosses above %D in the oversold region, suggesting a potential **uptrend**.
                - Bearish Signal: When %K crosses below %D in the overbought region, suggesting a potential **downtrend**.
            """)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")