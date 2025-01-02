import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import accuracy_score, confusion_matrix


def prepare_time_series_data(data, column='Close', sequence_length=10):
    """
    Prepare time-series data for CNN.
    """
    data['Target'] = (data[column].shift(-1) > data[column]).astype(int)
    features, labels = [], []

    for i in range(len(data) - sequence_length):
        features.append(data[column].iloc[i:i + sequence_length].values)
        labels.append(data['Target'].iloc[i + sequence_length])

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def cnn_model():
    """
    CNN for predicting next day's price movement based on time-series data.
    """
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return

    st.title("CNN for Time-Series Price Movement Prediction")
    data = st.session_state['data'].copy()

    # Normalize the data
    scaler = MinMaxScaler()
    data['Close'] = scaler.fit_transform(data[['Close']])

    # Prepare time-series data
    sequence_length = st.slider("Sequence Length (days)", min_value=5, max_value=50, value=10, step=1)
    features, labels = prepare_time_series_data(data, sequence_length=sequence_length)

    # Reshape features for CNN
    features = features.reshape(features.shape[0], features.shape[1], 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build CNN model
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(sequence_length, 1)),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    if st.button("Train CNN Model"):
        with st.spinner("Training CNN model..."):
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Evaluate the model
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.success(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # Predict next day's movement
        st.subheader("Next Day Prediction")
        last_sequence = features[-1].reshape(1, sequence_length, 1)
        prediction = model.predict(last_sequence)
        st.write("Predicted Movement: Up" if prediction[0][0] > 0.5 else "Predicted Movement: Down")
