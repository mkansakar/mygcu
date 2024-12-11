import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_features(data, time_steps=60):
    """
    Creates lagged features for LSTM.
    
    Args:
        data (ndarray): Scaled data array.
        time_steps (int): Number of time steps for lagging.

    Returns:
        X, y: Feature and target arrays.
    """
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def lstm_model():
    """
    LSTM model for stock price prediction with scaling, lagging, and cross-validation.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("No data found. Please load data first.")
        return

    st.title("LSTM Model for Stock Price Prediction")

    # Load data
    data = st.session_state['data']['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Lagged feature generation
    time_steps = st.slider("Select Time Steps for LSTM", 30, 120, 60, key="lstm_time_steps")
    X, y = create_lstm_features(scaled_data, time_steps)

    # Time-series cross-validation setup
    tscv = TimeSeriesSplit(n_splits=3)
    fold = 1
    metrics = []

    for train_index, test_index in tscv.split(X):
        st.write(f"Training fold {fold}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshape data for LSTM (samples, time_steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        epochs = st.slider("Select Epochs", 10, 100, 50, key=f"lstm_epochs_fold_{fold}")
        batch_size = st.slider("Select Batch Size", 16, 128, 32, key=f"lstm_batch_fold_{fold}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predict and evaluate
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_test, predictions)
        metrics.append(mse)
        st.write(f"Fold {fold} - Mean Squared Error: {mse:.4f}")
        fold += 1

    # Display average cross-validation accuracy
    avg_mse = np.mean(metrics)
    st.write(f"Average MSE across folds: {avg_mse:.4f}")

    # Full training on entire dataset
    st.subheader("Final Training and Prediction")
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Final Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Final Evaluation
    mse = mean_squared_error(y_test, predictions)
    st.write(f"Final Model MSE: {mse:.4f}")

    # Plot results
    st.subheader("Actual vs Predicted Prices")
    st.line_chart(pd.DataFrame({"Actual": y_test.flatten(), "Predicted": predictions.flatten()}))

    # Save the model
    if st.button("Save LSTM Model"):
        model.save("lstm_model.h5")
        st.success("LSTM model saved as `lstm_model.h5`!")
