import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data, split_data

def svm_model():
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left")
        return
    
    st.title("SVM Price Movement")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    
    data = preprocess_data(st.session_state['data'].copy())
    features, target, splits = split_data(data)
    
    for train_idx, test_idx in splits:
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    
    model = SVC()
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


    with st.expander("What is Support Vector Machine?"):
        st.write("""
            Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. In stock price movement prediction, SVM is used as a classifier to determine whether a stock will go up or down based on historical and technical data.\n
            Interpretation & Reliability:\n
                Higher Accuracy (close to 1 or 100%) → The model makes very few mistakes.\n
                Lower Accuracy → The model struggles to differentiate between classes.\n
                High Precision → Few false positives (FP), meaning when the model predicts positive, it is usually correct.\n
                Low Precision → Many false positives, meaning the model frequently predicts positive incorrectly.\n
                High Recall → Few false negatives (FN), meaning the model captures most of the actual positive cases.\n
                Low Recall → Many false negatives, meaning the model misses too many actual positive cases.\n
                High F1-Score → The model is good at both Precision and Recall.\n
                Low F1-Score → The model either has a low Precision or Recall (or both).
        """)