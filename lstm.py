import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from data_preprocessing import preprocess_data, split_data

def prepare_sequences(features, target, sequence_length=30):
    """
    Convert stock data into sequences for LSTM training.
    """
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Predicts movement (UP/DOWN)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model():
    """
    Train LSTM using Time Series Cross-Validation and predict next-day movement.
    """
    st.title("LSTM Stock Price Movement Prediction")

    if 'data' not in st.session_state:
        st.error("Please load stock data first.")
        return

    # Load and preprocess data
    features, target, _ = split_data(st.session_state['filtered_features'])

    # Convert data to sequences
    sequence_length = 30
    X, y = prepare_sequences(features, target, sequence_length)

    # Perform Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build and train model
        model = build_lstm_model((sequence_length, X.shape[2]))
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)  # Reduce epochs for faster training

        # Make predictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        # Compute performance metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    # Average metrics across folds
    accuracy_mean = np.mean(accuracy_scores) * 100
    precision_mean = np.mean(precision_scores)
    recall_mean = np.mean(recall_scores)
    f1_mean = np.mean(f1_scores)

    # Predict next-day movement
    last_sequence = X[-1].reshape(1, sequence_length, X.shape[2])
    predicted_movement = model.predict(last_sequence)[0][0]
    movement = "Up" if predicted_movement > 0.5 else "Down"

    # Display Model Performance Metrics
    st.subheader("Model Performance (Cross-Validation):")
    st.write(f"**Accuracy:** {accuracy_mean:.2f}%")
    st.write(f"**Precision:** {precision_mean:.2f}")
    st.write(f"**Recall:** {recall_mean:.2f}")
    st.write(f"**F1 Score:** {f1_mean:.2f}")

    # Display Next-Day Prediction
    st.subheader("Next Day Prediction:")
    st.write(f"**Next Day Price Movement:** {movement}")

