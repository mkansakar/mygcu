#logistic_regression.py
from data_preprocessing import preprocess_data, split_data
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


def logistic_regression():
    try:
        if 'filtered_features' not in st.session_state or st.session_state['filtered_features'] is None:
            st.error("Please proceed with Data Preprocessing to load the preprocessed data.")
            return
        
        st.title("Logistic Regression Price Movement")
        st.markdown(f"Stock: {st.session_state['symbol']}")
        
        # Preprocess data
        #st.dataframe(st.session_state['filtered_features'].tail(2))
        #proc_data = st.session_state['filtered_features'].copy()



        features, target, _ = split_data(st.session_state['filtered_features']) # Ensure `split_data()` returns clean X, y

        features = pd.DataFrame(features)  # Convert features to DataFrame
        target = pd.Series(target).reset_index(drop=True)  # Convert target to Pandas Series   

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)  # Define 5 folds for time-series cross-validation
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        try: 

            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Compute metrics
                accuracies.append(accuracy_score(y_test, predictions))
                precisions.append(precision_score(y_test, predictions))
                recalls.append(recall_score(y_test, predictions))
                f1_scores.append(f1_score(y_test, predictions))
        except Exception as train_error:
            raise RuntimeError(f"Model Training Error: {train_error}")       
        
        
        # Display averaged performance metrics
        st.write("__Model Performance Across Time-Series Splits__:")    
        st.write(f"Accuracy: {np.mean(accuracies) * 100:.2f}%")
        st.write(f"Precision: {np.mean(precisions):.2f}")
        st.write(f"Recall: {np.mean(recalls):.2f}")
        st.write(f"F1 Score: {np.mean(f1_scores):.2f}")
        
        # Predict next day's movement
        st.write("__Next Day Prediction__:")    
        last_row = features.iloc[[-1]]  # Corrected reshape for single sample
        prediction = model.predict(last_row)
        st.write("Next Day Price Movement: **Up**" if prediction[0] == 1 else "Next Day Price Movement: **Down**")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    #st.write(st.session_state['filtered_features'].tail(2))

    with st.expander("What is Logistic Regression?"):
        st.write("""
            Logistic Regression is a statistical model used for binary classification, meaning it predicts one of two possible outcomes. In the case of stock price movement, it is commonly used to predict whether a stock will go up or down on the next trading day.\n
            Interpretation & Reliability:\n
                The low accuracy means the model struggles with predicting price movements accurately.\n
                Since stock price movements are inherently noisy and difficult to predict, Logistic Regression might not be the best model for this task. \n
                Real-world trading decisions should not rely solely on this prediction, as the model’s poor performance indicates high uncertainty.     
        """)
