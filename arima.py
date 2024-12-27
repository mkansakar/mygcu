import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import plotly.graph_objects as go

def check_stationarity(data, column):
    """
    Perform the Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(data[column].dropna())
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])
    st.write("Critical Values:", result[4])
    return result[1] <= 0.05  # Stationary if p-value <= 0.05

def arima_model():
    """
    Apply the ARIMA model to predict stock prices with time-series cross-validation.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("ARIMA Model for Stock Price Prediction")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    # Load the data
    data = st.session_state['data']
    column = st.selectbox("Select a column for ARIMA modeling", data.columns, index=data.columns.get_loc("Close"))

    st.subheader("Step 1: Check Stationarity")
    if st.checkbox("Check Stationarity"):
        is_stationary = check_stationarity(data, column)
        if is_stationary:
            st.success("The data is stationary.")
        else:
            st.warning("The data is not stationary. Consider applying transformations.")

    # Apply transformations if needed
    st.subheader("Step 2: Apply Transformations (Optional)")
    transformed_data = data[column].copy()
    if st.checkbox("Apply Differencing"):
        transformed_data = transformed_data.diff().dropna()
        st.write("Differenced Data")
        st.line_chart(transformed_data)

    if st.checkbox("Apply Log Transformation"):
        if (data[column] <= 0).any():
            st.warning(f"Log transformation cannot be applied to column '{column}' because it contains non-positive values.")
        else:
            transformed_data = np.log(transformed_data)
            st.write("Log Transformed Data")
            st.line_chart(transformed_data)

    # ARIMA modeling
    st.subheader("Step 3: Configure ARIMA Parameters")
    p = st.number_input("Select p (AR order)", min_value=0, max_value=5, value=1, step=1)
    d = st.number_input("Select d (Differencing order)", min_value=0, max_value=2, value=1, step=1)
    q = st.number_input("Select q (MA order)", min_value=0, max_value=5, value=1, step=1)

    if st.button("Train ARIMA Model with Cross-Validation"):
        try:
            # Time-series cross-validation
            st.subheader("Step 4: Time-Series Cross-Validation")
            ts_split = TimeSeriesSplit(n_splits=3)
            errors = []

            for train_index, test_index in ts_split.split(transformed_data):
                train, test = transformed_data.iloc[train_index], transformed_data.iloc[test_index]

                # Fit ARIMA model
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()

                # Forecast on the test set
                forecast = model_fit.forecast(steps=len(test))
                error = mean_squared_error(test, forecast)
                errors.append(error)

            avg_error = np.mean(errors)
            st.write(f"Cross-Validation MSE: {avg_error:.2f}")

            # Final model fit
            st.subheader("Step 5: Final Model Fit and Forecast")
            final_model = ARIMA(transformed_data, order=(p, d, q))
            final_model_fit = final_model.fit()

            # Forecast future values
            forecast_steps = st.slider("Select number of steps to forecast", min_value=1, max_value=30, value=20, step=1)
            forecast = final_model_fit.forecast(steps=forecast_steps)
            forecast_index = pd.date_range(start=transformed_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"])

            # Plot the forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=transformed_data.index, y=transformed_data, mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Forecast"], mode='lines', name='Forecast', line=dict(color='red')))
            fig.update_layout(title="ARIMA Forecast", xaxis_title="Date", yaxis_title="Value", template="plotly_white")
            st.plotly_chart(fig)

            # Display model accuracy
            st.success(f"ARIMA model trained successfully with final accuracy (MSE): {avg_error:.2f}")

            # Store results
            st.session_state['arima_forecast'] = forecast_df
        except Exception as e:
            st.error(f"An error occurred during ARIMA modeling: {e}")
