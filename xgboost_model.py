# xgboost_model.py
import streamlit as st
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import split_data

def train_xgboost():
    """
    Train XGBoost using Time Series Cross-Validation and predict next-day movement.
    Includes exception handling to prevent crashes.
    """
    try:
        # Ensure preprocessed data is available
        if 'filtered_features' not in st.session_state or st.session_state['filtered_features'] is None:
            st.error("Please proceed with Data Preprocessing to load the preprocessed data.")
            return

        st.title("XGBoost Price Movement Prediction")
        st.markdown(f"Stock: {st.session_state.get('symbol', 'Unknown')}")

        # Prepare data
        try:
            features, target, _ = split_data(st.session_state['filtered_features'])
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return
        
        # Define XGBoost Model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        try:
            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Store performance metrics
                metric_funcs = [accuracy_score, precision_score, recall_score, f1_score]
                for key, func in zip(metrics.keys(), metric_funcs):
                    metrics[key].append(func(y_test, y_pred))
        except Exception as e:
            st.error(f"Error during model training and evaluation: {e}")
            return

        # Compute and display average scores
        avg_metrics = {key: np.mean(value) for key, value in metrics.items()}

        st.subheader("Model Performance (Cross-Validation)")
        st.write(f"**Accuracy:** {avg_metrics['accuracy'] * 100:.2f}%")
        st.write(f"**Precision:** {avg_metrics['precision']:.2f}")
        st.write(f"**Recall:** {avg_metrics['recall']:.2f}")
        st.write(f"**F1 Score:** {avg_metrics['f1']:.2f}")

        # Predict next-day movement
        try:
            last_row = features[-1].reshape(1, -1)
            prediction = model.predict(last_row)[0]
            movement = "Up" if prediction == 1 else "Down"
            #st.write("__Next Day Prediction__:") 
            st.write(f"**Next Day Price Movement:** {movement}")
        except Exception as e:
            st.error(f"Error making next-day prediction: {e}")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    with st.expander("What is XGBoost?"):
        st.write("""
            Boosting Method: Instead of training multiple models independently, XGBoost trains models sequentially, where each new model corrects the errors of the previous one.
Weighted Learning: Each misclassified instance gets higher weight, making the model more focused on difficult cases.
            
            How Does XGBoost Work?
                Boosting Method: Instead of training multiple models independently, XGBoost trains models sequentially, where each new model corrects the errors of the previous one.
                Weighted Learning: Each misclassified instance gets higher weight, making the model more focused on difficult cases.
                Tree-Based Learning: Uses decision trees to make predictions by splitting data at critical points
        """)
