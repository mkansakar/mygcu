import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def svm_model():
    st.title("Support Vector Machine for Next Day Price Movement")
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first.")
        return

    data = st.session_state['data']
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = data.drop(['Target', 'Close'], axis=1).dropna()
    target = data['Target'].dropna()

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text(classification_report(y_test, predictions))
