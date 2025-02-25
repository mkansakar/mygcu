# model_comparison.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import preprocess_data, split_data
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from data_preprocessing import preprocess_data, split_data


def calculate_rsi(data, column='Close', window=14):
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_atr(data, high='High', low='Low', close='Close', window=14):
    high_low = data[high] - data[low]
    high_close = (data[high] - data[close].shift()).abs()
    low_close = (data[low] - data[close].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_cmf(data, high='High', low='Low', close='Close', volume='Volume', window=20):
    money_flow_multiplier = ((data[close] - data[low]) - (data[high] - data[close])) / (data[high] - data[low])
    money_flow_volume = money_flow_multiplier * data[volume]
    cmf = money_flow_volume.rolling(window=window).sum() / data[volume].rolling(window=window).sum()
    return cmf

def compare_models():
    try:
        if 'filtered_features' not in st.session_state or st.session_state['filtered_features'] is None:
            st.error("Please proceed with Data Preprocessing to load the preprocessed data.")
            return
        
        st.title("Model Comparison")
        st.markdown(f"Stock: {st.session_state['symbol']}")

        features, target, _ = split_data(st.session_state['filtered_features'])
        
        # Add Up/Down Trend Column
        trend = np.where(target.diff() > 0, 'Up', 'Down')
        trend[0] = 'N/A'  # First value has no previous value to compare
        
        # Define individual models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', C=1e3, probability=True, gamma=0.1, random_state=42)
        }
        
        # Train-Test Split for Model Evaluation
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

        # Train Individual Models
        for model_name, model in models.items():
            model.fit(X_train, y_train)

        # Compute Individual Model Accuracies
        accuracies = {
            model_name: accuracy_score(y_test, model.predict(X_test))
            for model_name, model in models.items()
        }

        # Normalize Accuracies to Assign Weights
        total_acc = sum(accuracies.values())
        weights = [accuracies[m] / total_acc for m in models.keys()]

        # Create Weighted Voting Classifier and Fit it
        ensemble_model = VotingClassifier(
            estimators=[(name, models[name]) for name in models.keys()],
            voting="soft",
            weights=weights
        )
        
        ensemble_model.fit(X_train, y_train)  # Fit the ensemble model
        
        models["Modified Voting Ensemble"] = ensemble_model

        # Train and Evaluate Models
        results = []
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in models.items():
            accuracy_scores = cross_val_score(model, features, target, cv=tscv, scoring='accuracy')
            precision_scores = cross_val_score(model, features, target, cv=tscv, scoring='precision')
            recall_scores = cross_val_score(model, features, target, cv=tscv, scoring='recall')
            f1_scores = cross_val_score(model, features, target, cv=tscv, scoring='f1')
            
            results.append({
                "Model": model_name,
                "Accuracy": round(np.mean(accuracy_scores), 2),
                "Precision": round(np.mean(precision_scores), 2),
                "Recall": round(np.mean(recall_scores), 2),
                "F1 Score": round(np.mean(f1_scores), 2)
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Display results
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df)
        
        # Next Day Prediction
        st.write("__Next Day Prediction Using Ensemble Method__:")
        last_row = features[-1].reshape(1, -1)
        prediction = ensemble_model.predict(last_row)
        st.write("Next Day Price Movement: **Up**" if prediction[0] == 1 else "Next Day Price Movement: **Down**")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    with st.expander("About Model Performance Comparison"):
            st.write("""
            Model performance comparison is a critical step in evaluating and selecting the best predictive model for a given task. The goal is to assess how different machine learning models perform against each other using key evaluation metrics such as accuracy, precision, recall, and F1-score. These metrics help determine which model provides the most reliable predictions while balancing trade-offs between false positives and false negatives.
            - Accuracy – Measures the proportion of correctly classified instances out of all predictions. Higher accuracy suggests better overall model performance, but it can be misleading for imbalanced datasets.
            - Precision – Represents the proportion of true positive predictions among all predicted positives. High precision indicates fewer false positives, which is critical in applications where false alarms must be minimized.
            - Recall (Sensitivity) – Measures how well the model identifies actual positive cases. High recall is crucial in situations where missing a positive case (false negative) is costly.
            - F1-score – A harmonic mean of precision and recall. It provides a balanced measure, especially when dealing with imbalanced classes.
                     
            """) 