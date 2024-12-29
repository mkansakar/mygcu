import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Function to calculate technical indicators
def add_technical_indicators(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['Volatility'] = data['Close'].rolling(window=10).std()
    return data

# Function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Main function for the Streamlit app
def svm_model():
    st.title("Support Vector Machine for Next Day Price Movement")

    # Check if data is loaded in session state
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first.")
        return

    # Load data from session state
    data = st.session_state['data']

    # Add technical indicators
    data = add_technical_indicators(data)

    # Generate the target for next-day movement
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Drop rows with NaN values due to shifting and indicator calculations
    data = data.dropna()

    # Prepare features and target
    features = data.drop(['Target', 'Close'], axis=1)
    target = data['Target']

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

    # Train the SVM model
    model = SVC()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text(classification_report(y_test, predictions))

    # Predict the next day's movement
    last_row = scaled_features[[-1]]  # Use the last available row for prediction
    next_day_prediction = model.predict(last_row)[0]

    # Display the prediction result
    movement = "UP" if next_day_prediction == 1 else "DOWN"
    st.write(f"Predicted next day's movement: **{movement}**")
