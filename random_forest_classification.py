# random_forest_classification.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import preprocess_data, split_data

def random_forest_classification():
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left")
        return
    
    st.title("Random Forest Price Movement")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    # Preprocess Data
    data = preprocess_data(st.session_state['data'].copy())
    features, target, splits = split_data(data)

    # Define Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Implement Cross-Validation (Time Series Split)
    tscv = TimeSeriesSplit(n_splits=5)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Store Performance Metrics
        accuracy_scores.append(accuracy_score(y_test, predictions))
        precision_scores.append(precision_score(y_test, predictions))
        recall_scores.append(recall_score(y_test, predictions))
        f1_scores.append(f1_score(y_test, predictions))

    # Compute Average Scores
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    # Display Results
    st.write(f"__Model Performance (Cross-Validation)__:")
    st.write(f"Accuracy: {avg_accuracy * 100:.2f}%")
    st.write(f"Precision: {avg_precision:.2f}")
    st.write(f"Recall: {avg_recall:.2f}")
    st.write(f"F1 Score: {avg_f1:.2f}")

    # Next-Day Prediction
    st.write("__Next Day Prediction__:")    
    last_row = features[-1].reshape(1, -1)
    prediction = model.predict(last_row)
    st.write("Next Day Price Movement: **Up**" if prediction[0] == 1 else "Next Day Price Movement: **Down**")

    with st.expander("What is Random Forest?"):
        st.write("""
            Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and robustness. It is commonly used for stock price movement classification, where the objective is to predict if a stock price will increase (Up), decrease (Down), or remain neutral based on historical market data and technical indicators.\n
            Interpretation & Reliability:\n
                - **Higher Accuracy** → Model correctly predicts stock movements.\n
                - **High Precision** → Few false positives (wrong buy signals).\n
                - **High Recall** → Few false negatives (missed buying opportunities).\n
                - **High F1-Score** → Balances Precision and Recall for imbalanced datasets.
        """)

