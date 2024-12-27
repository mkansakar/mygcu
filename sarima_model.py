import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def sarima_forecast(data, column="Close", seasonal_order=(1, 1, 1, 12), forecast_periods=30):
    """
    Train a SARIMA model and forecast future stock prices.
    """
    try:
        st.subheader("SARIMA Forecasting")
        st.markdown(f"Stock: {st.session_state['symbol']}")
        # Select column to forecast
        st.write(f"Training SARIMA model on column: **{column}**")
        ts_data = data[column].dropna()

        # Log transformation (optional for stabilizing variance)
        st.write("Applying log transformation to stabilize variance.")
        ts_data_log = np.log(ts_data + 1)

        # Train SARIMA Model
        model = SARIMAX(ts_data_log, order=(1, 1, 1), seasonal_order=seasonal_order, enforce_stationarity=False)
        model_fit = model.fit(disp=False)

        st.success("SARIMA model trained successfully!")
        st.write(model_fit.summary())

        # Forecast future periods
        forecast = model_fit.forecast(steps=forecast_periods)
        forecast = np.exp(forecast) - 1  # Reverse the log transformation

        # Create Forecast Index
        forecast_index = pd.date_range(start=ts_data.index[-1], periods=forecast_periods + 1, freq="D")[1:]

        # Combine results
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"])

        # Plot Results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Actual Prices'))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Forecast"], mode='lines', name='SARIMA Forecast', line=dict(color='red')))
        fig.update_layout(title="SARIMA Forecast", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(fig)

        # Display Metrics (Optional)
        st.subheader("Model Accuracy")
        actual = ts_data[-forecast_periods:]  # Last known periods for comparison
        mse = mean_squared_error(actual[-len(forecast):], forecast[:len(actual)])
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")

        return forecast_df

    except Exception as e:
        st.error(f"Error during SARIMA forecasting: {e}")

def sarima_model():
    """
    Streamlit SARIMA Module Integration.
    """
    st.title("SARIMA Stock Price Forecasting")

    if "data" not in st.session_state or st.session_state["data"] is None:
        st.error("No data loaded. Please load stock price data first.")
        return

    # User Input for Parameters
    data = st.session_state["data"]
    column = st.selectbox("Select Column for SARIMA Forecasting", data.columns, index=data.columns.get_loc("Close"))
    seasonal_period = st.slider("Select Seasonal Period (e.g., 7 for weekly)", min_value=4, max_value=365, value=12)
    forecast_periods = st.slider("Number of Days to Forecast", min_value=7, max_value=60, value=30)

    # Run SARIMA Forecasting
    if st.button("Run SARIMA Forecast"):
        sarima_forecast(data, column=column, seasonal_order=(1, 1, 1, seasonal_period), forecast_periods=forecast_periods)
