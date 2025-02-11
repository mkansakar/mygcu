#spectral_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import periodogram
import plotly.graph_objects as go

def spectral_analysis():
    """
    Perform spectral analysis on stock prices to identify dominant frequencies.
    """

    if 'data' not in st.session_state:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Spectral Analysis of Stock Prices")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    # Load the data
    data = st.session_state['data']

    # Select column for analysis
    column = st.selectbox("Select a column for spectral analysis", data.columns, index=data.columns.get_loc("Close"))

    # Compute the power spectral density
    st.subheader("Frequency Domain Analysis")
    #detrend = st.checkbox("Apply Detrending")
    #normalize = st.checkbox("Normalize Data")   

    #data = data[column].dropna()
    #data = data.to_numpy()

    # if detrend:
    #     data = data - data.mean()
    # if normalize:
    #     data = (data - data.min()) / (data.max() - data.min())


    freq, power = periodogram(data, scaling='spectrum')

    # Create a DataFrame for visualization
    spectral_df = pd.DataFrame({"Frequency": freq, "Power": power})

    # Plot the power spectral density
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spectral_df["Frequency"], y=spectral_df["Power"], mode="lines", name="Power Spectrum"))
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency",
        yaxis_title="Power",
        template="plotly_white"
    )
    st.plotly_chart(fig)

    # Highlight dominant frequencies
    st.subheader("Dominant Frequencies")
    top_frequencies = spectral_df.nlargest(5, "Power")
    st.write(top_frequencies)

    with st.expander("What is Spectral Analysis?"):
        st.write("""
            Spectral analysis in stock price forecasting is a mathematical technique used to decompose a time-series signal into its frequency components. It helps identify periodic patterns, trends, and cyclical behaviors in the data by transforming the time-domain series into the frequency domain.\n 
            Dominant Frequencies: Peaks in the frequency spectrum correspond to dominant cycles in the stock price data.\n
                 A peak at a frequency corresponding to 1/12 indicates a monthly cycle.
            Low Frequencies: Represent long-term trends or cycles, such as seasonal effects or economic conditions.\n
            High Frequencies: Represent short-term fluctuations or market volatility.\n
        """)
