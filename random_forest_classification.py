import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def calculate_rsi(data, column='Close', window=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD and Signal Line.
    """
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_atr(data, high='High', low='Low', close='Close', window=14):
    """
    Calculate the Average True Range (ATR).
    """
    high_low = data[high] - data[low]
    high_close = (data[high] - data[close].shift()).abs()
    low_close = (data[low] - data[close].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_cmf(data, high='High', low='Low', close='Close', volume='Volume', window=20):
    """
    Calculate the Chaikin Money Flow (CMF).
    """
    money_flow_multiplier = ((data[close] - data[low]) - (data[high] - data[close])) / (data[high] - data[low])
    money_flow_volume = money_flow_multiplier * data[volume]
    cmf = money_flow_volume.rolling(window=window).sum() / data[volume].rolling(window=window).sum()
    return cmf

def random_forest_classification():
    """
    Random Forest for Stock Price Movement Prediction.
    """

    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please load the data first from the sidebar on the left")
        return

    st.title("Random Forest Price Movement")
    st.markdown(f"Stock: {st.session_state['symbol']}")

    data = st.session_state['data'].copy()

    # Add target variable
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Add Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Add RSI
    data['RSI'] = calculate_rsi(data)

    # Add MACD
    data['MACD'], data['Signal_Line'] = calculate_macd(data)

    # Add ATR
    data['ATR'] = calculate_atr(data)

    # Add CMF
    data['CMF'] = calculate_cmf(data)

    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['Daily_Return'] = data['Close'].pct_change()

    # Drop NaN values resulting from calculations
    data.dropna(inplace=True)

    # Define features and target
    features = data[['Close', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Signal_Line', 'Momentum', 'Daily_Return', 'ATR', 'CMF']]
    target = data['Target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Cross-validation scores
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
    st.write(f"Cross-Validation Accuracy Scores: {cross_val_scores}")
    st.write(f"Mean Cross-Validation Accuracy: {cross_val_scores.mean() * 100:.2f}%")

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, predictions))

    # Predict next day's movement
    st.subheader("Next Day Prediction")
    last_row = features.iloc[-1:].values
    prediction = model.predict(last_row)
    st.write("Next Day Price Movement: **Up**" if prediction[0] == 1 else "Next Day Price Movement: **Down**")