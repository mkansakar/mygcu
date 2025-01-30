import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from data_preprocessing import preprocess_data, split_data


def random_forest_classification():
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left")
        return
    
    st.title("Random Forest Price Movement")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    
    data = preprocess_data(st.session_state['data'].copy())
    features, target, splits = split_data(data)
    
    for train_idx, test_idx in splits:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    st.write(f"__Model Performance__:")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    
    # Next Day Prediction
    st.write("__Next Day Prediction__:")
    last_row = features[-1].reshape(1, -1)
    prediction = model.predict(last_row)
    st.write("Next Day Price Movement: **Up**" if prediction[0] == 1 else "Next Day Price Movement: **Down**")

    with st.expander("What is Random Forest?"):
        st.write("""
            Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and robustness. It is commonly used for stock price movement classification, where the objective is to predict if a stock price will increase (Up), decrease (Down), or remain neutral based on historical market data and technical indicators.\n
            Interpretation & Reliability:\n
                Accuracy - Measures the overall correctness of the model. High accuracy means the model predicts stock movements well.\n
                Precision - Measures how many predicted Up movements were actually correct. Useful when false positives (wrong buy signals) are costly.\n
                Recall - Measures how many actual Up movements were correctly identified. Important when missing buy opportunities is critical.\n
                F1-Score - Balances Precision and Recall. Useful in imbalanced datasets (e.g., more "Down" days than "Up").    
        """)