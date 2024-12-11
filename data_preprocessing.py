import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    """
    Data Preprocessing Module.
    """
    if 'data' in st.session_state:
        st.title("Data Preprocessing")

        # Load data
        data = st.session_state['data'].copy()

        # Missing Values Handling
        st.subheader("Handle Missing Values")
        if data.isnull().values.any():
            st.warning(f"Missing values detected! Total: {data.isnull().sum().sum()}")
            missing_method = st.radio(
                "Choose a method to handle missing values:",
                options=["Forward Fill", "Backward Fill", "Mean Replacement", "Remove Rows"]
            )
            if st.button("Apply Missing Value Handling"):
                if missing_method == "Forward Fill":
                    data.fillna(method="ffill", inplace=True)
                elif missing_method == "Backward Fill":
                    data.fillna(method="bfill", inplace=True)
                elif missing_method == "Mean Replacement":
                    data.fillna(data.mean(), inplace=True)
                elif missing_method == "Remove Rows":
                    data.dropna(inplace=True)
                st.success("Missing values handled successfully!")
                st.dataframe(data.head())
        else:
            st.success("No missing values found.")

        # Scaling Data
        st.subheader("Scale Data")
        columns_to_scale = st.multiselect(
            "Select columns to scale (Min-Max Scaling):",
            options=["Open", "High", "Low", "Close", "Volume"],
            default=["Open", "High", "Low", "Close", "Volume"]
        )
        if st.button("Apply Scaling"):
            scaler = MinMaxScaler()
            data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
            st.success("Data scaled successfully!")
            st.dataframe(data.head())

        # Feature Engineering
        st.subheader("Feature Engineering")
        if st.checkbox("Add Moving Averages"):
            ma_window = st.selectbox("Select Moving Average Window:", [5, 10, 20, 30])
            data[f"MA_{ma_window}"] = data['Close'].rolling(window=ma_window).mean()
            st.success(f"{ma_window}-day Moving Average added!")
            st.dataframe(data[[f"MA_{ma_window}"]].head())

        if st.checkbox("Add Relative Strength Index (RSI)"):
            period = st.number_input("RSI Period:", min_value=1, value=14)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            st.success("RSI added!")
            st.dataframe(data[['RSI']].head())

        if st.checkbox("Add MACD"):
            short_window = st.number_input("Short EMA Window:", min_value=1, value=12)
            long_window = st.number_input("Long EMA Window:", min_value=1, value=26)
            signal_window = st.number_input("Signal Line Window:", min_value=1, value=9)
            data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
            data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
            data['MACD'] = data['EMA_short'] - data['EMA_long']
            data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
            st.success("MACD and Signal Line added!")
            st.dataframe(data[['MACD', 'Signal_Line']].head())

        # Save Preprocessed Data
        if st.button("Save Preprocessed Data"):
            st.session_state['preprocessed_data'] = data
            st.success("Preprocessed data saved to session state!")
            st.dataframe(data.head())
    else:
        st.error("No data available. Please load data first.")
