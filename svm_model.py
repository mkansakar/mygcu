# svm_model.py
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data, split_data

def svm_model():
    try:

        if 'filtered_features' not in st.session_state or st.session_state['filtered_features'] is None:
            st.error("Please proceed with Data Preprocessing to load the preprocessed data.")
            return
        
        st.title("SVM Price Movement")
        st.markdown(f"Stock: {st.session_state['symbol']}")

        # Preprocess Data
        features, target, _ = split_data(st.session_state['filtered_features'])

        # Define SVM Model
        model = SVC(kernel='rbf', C=1e3, probability=True, gamma=0.1, random_state=42)

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

    # except Exception as e:
    #     st.error(f"An unexpected error occurred: {e}")

        with st.expander("What is Support Vector Machine?"):
            st.write("""
                Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. In stock price movement prediction, SVM is used as a classifier to determine whether a stock will go up or down based on historical and technical data.\n
                Interpretation & Reliability:\n
                    - Higher Accuracy - Fewer mistakes in predictions.\n
                    - High Precision - Few false positives (incorrectly predicting an "Up" movement).\n
                    - High Recall - Few false negatives (missing actual "Up" movements).\n
                    - High F1-Score - Good balance between Precision and Recall.
            """)

    except ValueError as ve:
        st.error(f"Value error in model training: {ve}. Ensure data is formatted correctly.")
    except Exception as e:
        st.error(f"An unexpected error occurred while training the model: {e}")
