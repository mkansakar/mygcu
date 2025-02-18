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

def compare_models():
    if 'filtered_features' not in st.session_state or st.session_state['filtered_features'] is None:
        st.error("Please proceed with Data Preprocessing to load the preproceed data.")
        return
    
    st.title("Model Comparison")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    #data = preprocess_data(st.session_state['data'].copy())
    #features, target, _ = split_data(data)
    features, target, _ = split_data(st.session_state['filtered_features'])
    
    # Define individual models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        #
        "SVM": SVC(kernel='rbf', C=1e3, probability=True, gamma=0.1, random_state=42)  # Enable probability for Voting Classifier
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

    # Create Weighted Voting Classifier
    ensemble_model = VotingClassifier(
        estimators=[(name, models[name]) for name in models.keys()],
        voting="soft",
        weights=weights  # Assign Weights Based on Model Performance
    )

    # Add Ensemble Model to Dictionary
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
            "F1 Score": round(np.mean(f1_scores), 2),
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    st.subheader("Model Performance Comparison")
    st.dataframe(results_df)


    with st.expander("About Model Performance Comparison"):
            st.write("""
            Model performance comparison is a critical step in evaluating and selecting the best predictive model for a given task. The goal is to assess how different machine learning models perform against each other using key evaluation metrics such as accuracy, precision, recall, and F1-score. These metrics help determine which model provides the most reliable predictions while balancing trade-offs between false positives and false negatives.
            - Accuracy – Measures the proportion of correctly classified instances out of all predictions. Higher accuracy suggests better overall model performance, but it can be misleading for imbalanced datasets.
            - Precision – Represents the proportion of true positive predictions among all predicted positives. High precision indicates fewer false positives, which is critical in applications where false alarms must be minimized.
            - Recall (Sensitivity) – Measures how well the model identifies actual positive cases. High recall is crucial in situations where missing a positive case (false negative) is costly.
            - F1-score – A harmonic mean of precision and recall. It provides a balanced measure, especially when dealing with imbalanced classes.
                     
            """) 