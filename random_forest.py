import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def add_features(data, column, lags=5, rolling_window=10):
    """
    Add lagged features, rolling statistics, and trends to the dataset.

    Args:
        data (DataFrame): The input time-series data.
        column (str): The target column to base features on.
        lags (int): The number of lagged features to add.
        rolling_window (int): The window size for rolling statistics.

    Returns:
        DataFrame: The dataset with additional features.
    """
    df = data.copy()

    # Add lagged features
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[column].shift(lag)

    # Add rolling statistics
    df[f'rolling_mean_{rolling_window}'] = df[column].rolling(window=rolling_window).mean()
    df[f'rolling_std_{rolling_window}'] = df[column].rolling(window=rolling_window).std()

    # Add trends (difference with lag)
    df[f'trend_{rolling_window}'] = df[column] - df[column].shift(rolling_window)

    return df.dropna()

def random_forest_model():
    """
    Random Forests for Stock Price Prediction with lagged features, rolling statistics, and trends.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("Random Forest Model with Advanced Features")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    # Load the data
    data = st.session_state['data']
    column = st.selectbox("Select the target column for prediction", data.columns, index=data.columns.get_loc("Close"))

    # Add features to the dataset
    st.subheader("Feature Engineering")
    lags = st.slider("Number of Lagged Features", min_value=1, max_value=10, value=5)
    rolling_window = st.slider("Rolling Window Size", min_value=5, max_value=30, value=10)
    feature_data = add_features(data, column, lags=lags, rolling_window=rolling_window)

    # Display feature data
    st.write("Sample Feature Data:")
    st.dataframe(feature_data.head())

    # Split the data into training and testing sets
    X = feature_data.drop(columns=[column])
    y = feature_data[column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    st.subheader("Train the Random Forest Model")
    n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100)
    max_depth = st.slider("Maximum Depth of Trees", min_value=1, max_value=20, value=10)

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Model Accuracy (R² Score): {r2:.2%}")

    # Visualize the results
    st.subheader("Actual vs Predicted Prices")
    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}, index=y_test.index)
    st.line_chart(results_df)

    # Save the model
    if st.button("Save Model"):
        import joblib
        joblib.dump(rf, "random_forest_model.pkl")
        st.success("Random Forest model saved as `random_forest_model.pkl`!")
