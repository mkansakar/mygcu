import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier



def train_xgboost(X, y):
    """
    Train an XGBoost classifier and return the model, accuracy, and confusion matrix.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, conf_matrix

def xgboost_prediction():
    """
    Streamlit interface for XGBoost-based price movement prediction.
    """
   
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left.")
        return
    
    st.title("XGBoost Price Movement")
    st.markdown(f"Stock: {st.session_state['symbol']}")
    data = st.session_state['data'].copy()

    # Prepare data
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    for lag in range(1, 6):
        data[f'lag_{lag}'] = data['Close'].shift(lag)
    data.dropna(inplace=True)

    X = data.drop(columns=['Target'])
    y = data['Target']

    # Train model
    if st.button("Train XGBoost Model"):
        with st.spinner("Training XGBoost model..."):
            model, accuracy, conf_matrix = train_xgboost(X, y)

        st.success(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # Predict next day's movement
        st.subheader("Next Day Prediction")
        last_row = X.iloc[-1:]  # Ensure it has feature names
        prediction = model.predict(last_row)
        st.write("Predicted Movement: Up" if prediction[0] == 1 else "Predicted Movement: Down")

