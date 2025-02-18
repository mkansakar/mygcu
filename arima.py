# arima.py
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def check_stationarity(data, column):
    """
    Perform the Augmented Dickey-Fuller test to check stationarity.
    """
    
    result = adfuller(data[column].dropna())
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])
    st.write("Critical Values:", result[4])
    return result[1] <= 0.05  # Stationary if p-value <= 0.05

def grid_search_arima(data, p_values, d_values, q_values):
    """
    Perform grid search for the best ARIMA parameters.
    """
    best_score, best_cfg = float("inf"), None
    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_score:
                best_score, best_cfg = aic, (p, d, q)
        except Exception:
            continue
    return best_cfg, best_score

def arima_model():
    """
    Apply the ARIMA model to predict stock prices with enhanced features, including automatic parameter selection.
    """
    if 'session_data' not in st.session_state or st.session_state['session_data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("ARIMA Model for Stock Price Prediction")
    st.markdown(f"Stock: {st.session_state.get('symbol', 'Unknown')}")
    
    # Load the data
    scaler = MinMaxScaler(feature_range=(50, 500))
    
    data = st.session_state['session_data'].copy()
    data["Volume"] = scaler.fit_transform(data[["Volume"]])
    #st.write(data.tail(1)) 
    column = st.selectbox("Select a column for ARIMA modeling", data.columns, index=data.columns.get_loc("Close"))

    st.subheader("Step 1: Check Stationarity")
    if st.checkbox("Check Stationarity"):
        is_stationary = check_stationarity(data, column)
        if is_stationary:
            st.success("The data is stationary.")
        else:
            st.warning("The data is not stationary. Consider applying transformations.")

    # Apply transformations
    st.subheader("Step 2: Apply Transformations")
    transformed_data = data[column].copy()
    if not isinstance(transformed_data.index, pd.DatetimeIndex):
        st.error("Time series data must have a datetime index.")
    if st.checkbox("Apply Differencing"):
        transformed_data = transformed_data.diff().dropna()
        st.write("Differenced Data")
        st.line_chart(transformed_data)
        #st.write(transformed_data.tail(1))

    if st.checkbox("Apply Log Transformation"):
        if (data[column] <= 0).any():
            st.warning(f"Log transformation cannot be applied to column '{column}' because it contains non-positive values.")
        else:
            transformed_data = np.log(transformed_data)
            st.write("Log Transformed Data")
            st.line_chart(transformed_data)

    # ARIMA modeling
    st.subheader("Step 3: Configure ARIMA Parameters")
    use_auto_arima = st.checkbox("Use Auto ARIMA (Recommended)", value=True)

    if use_auto_arima:
        st.write("Finding the best parameters...")
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        best_cfg, best_aic = grid_search_arima(transformed_data, p_values, d_values, q_values)
        if best_cfg:
            st.success(f"Best ARIMA parameters: p={best_cfg[0]}, d={best_cfg[1]}, q={best_cfg[2]} (AIC: {best_aic:.2f})")
            p, d, q = best_cfg
        else:
            st.error("No suitable ARIMA parameters found.")
            p, d, q = 1, 1, 1
    else:
        p = st.number_input("Select p (AR order)", min_value=0, max_value=5, value=1, step=1)
        d = st.number_input("Select d (Differencing order)", min_value=0, max_value=2, value=1, step=1)
        q = st.number_input("Select q (MA order)", min_value=0, max_value=5, value=1, step=1)

    if st.button("Train ARIMA Model"):
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
            if forecast.empty:
                st.error("Forecast is empty. Verify the model and input data.")
            forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='D')[1:]
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"])

            # Display forecast table
            st.subheader("Forecast Results")
            st.dataframe(forecast_df)

            # Download forecast
            csv = forecast_df.to_csv(index=True)
            st.download_button("Download Forecast as CSV", csv, "forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"An error occurred during ARIMA modeling: {e}")
