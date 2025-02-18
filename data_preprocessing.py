import streamlit as st
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

# Technical Indicator Functions
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

def calculate_bollinger_bands(data, column='Close', window=20, std_dev=2):
    sma = data[column].rolling(window=window).mean()
    std = data[column].rolling(window=window).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    band_width = (upper_band - lower_band) / sma  
    return upper_band, lower_band, band_width

def compute_shap_feature_importance(features, target):
    """Compute SHAP feature importance using LightGBM."""
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    X_test = X_test[X_train.columns]
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with column mean (or use a more robust imputation method)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_train.mean(), inplace=True) 

    # Train LightGBM Model
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Compute SHAP Values
  
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test,check_additivity=False)

    # Compute Average Feature Importance
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': np.abs(shap_values.values).mean(axis=0)})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df

def preprocess_data():
    try:

        """Preprocess stock data and display feature selection UI."""
        
        if 'session_data' not in st.session_state or st.session_state['session_data'] is None:
            st.error("Please load the data first from the sidebar on the left.")
            return
        
        st.title("Data Preprocessing")
        st.markdown(f"**Stock: {st.session_state['symbol']}**")

        #Copy Data & Compute Technical Indicators
        data = st.session_state['session_data'].copy()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['Signal_Line'] = calculate_macd(data)
        data['ATR'] = calculate_atr(data)
        data['CMF'] = calculate_cmf(data)
        data['BB_Upper'], data['BB_Lower'], data['BB_Width'] = calculate_bollinger_bands(data)
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        data['Daily_Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)

        #Define Features & Target
        features = data[['Close', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Signal_Line', 'Momentum', 'Daily_Return',
                        'ATR', 'CMF', 'BB_Upper', 'BB_Lower', 'BB_Width', 'Gold_Close', 'GBPUSD']]
        target = (data['Close'].shift(-1) > data['Close']).astype(int)

        # Calculate the ratio of 0s and 1s
        target_counts = target.value_counts()
        total = len(target)

        # Compute ratios
        ratio_0 = target_counts[0] / total
        ratio_1 = target_counts[1] / total

        #Store Initial Features & Target in Session State
        st.session_state['features'] = features
        st.session_state['target'] = target

        #Display Latest 5 Rows
        # st.subheader("Sample Data (First Few Rows)")
        # st.dataframe(features.tail(2))

        # Compute SHAP Feature Importance
        shap_importance = compute_shap_feature_importance(features, target)

        # Display Feature Importance Table
        st.subheader("Feature Importance (SHAP) Using LightGBM")
        st.dataframe(shap_importance)

        # Button to Remove Low-Importance Features
        if st.button("Remove Less Important Features"):
            threshold = shap_importance['Importance'].median()  
            selected_features = shap_importance[shap_importance['Importance'] >= threshold]['Feature'].tolist()
            filtered_features = features[selected_features]

            # Store Filtered Features in Session State
            st.session_state['filtered_features'] = filtered_features
            st.success("Less important features removed!")

        # Display Updated Feature Set if Features Were Removed
        #if 'filtered_features' not in st.session_state or st.session_state['filtered_features'] is None:
        if 'filtered_features' in st.session_state and st.session_state['filtered_features'] is not None:

            st.subheader("Updated Feature Set (After Removal)")
            st.dataframe(st.session_state['filtered_features'].tail(2))

            st.write(f"Ratio of 0s (Price Down): {ratio_0:.2f} ({target_counts[0]} occurrences)")
            st.write(f"Ratio of 1s (Price Up): {ratio_1:.2f} ({target_counts[1]} occurrences)")


    except KeyError as ke:
        st.error(f"Missing data column: {ke}. Ensure the dataset contains 'Close' prices.")
    except Exception as e:
        st.error(f"An unexpected error occurred during data preprocessing: {e}")


def split_data(proc_data):
    """Split data into training and testing sets."""
    if 'filtered_features' not in st.session_state or 'target' not in st.session_state:
        st.error("Please preprocess data first.")
        return None, None, None
    
    features = st.session_state['filtered_features']
    target = st.session_state['target']

    # Scale Features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Time-Series Cross-Validation Splitting
    tscv = TimeSeriesSplit(n_splits=5)
    splits = [(train_idx, test_idx) for train_idx, test_idx in tscv.split(features_scaled)]

    return features_scaled, target, splits
