# model_comparison.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import preprocess_data, split_data
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models():
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left")
        return
    
    st.title("Model Comparison")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    data = preprocess_data(st.session_state['data'].copy())
    features, target, _ = split_data(data)
    
    # Define individual models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),

        "SVM": SVC(probability=True)  # Enable probability for Voting Classifier
    }
    
    # Ensemble: Voting Classifier (Soft Voting)
    ensemble_model = VotingClassifier(
        estimators=[
            ("LogReg", models["Logistic Regression"]),
            ("RandomForest", models["Random Forest"]),

            ("SVM", models["SVM"])
        ],
        voting="soft"  # Soft voting averages probability scores
    )
    
    models["Voting Classifier (Ensemble)"] = ensemble_model

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

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    st.pyplot(plt)
