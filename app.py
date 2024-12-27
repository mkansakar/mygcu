# app.py
import streamlit as st
from login import login
from contact_us import contact_us
from registration import register_user
from data_load import load_data
from data_visualization import visualize_data
from price_return import calculate_price_returns
from moving import moving_indicators
from trend_slope import compute_trend_slope
from decompose import decompose_time_series
#from resampling import resample_data
from volatility_indicators import display_volatility_indicators
from data_preprocessing import preprocess_data
#from check_stationarity import check_stationarity
#from transformation import apply_transformations
from arima import arima_model
from random_forest import random_forest_model
#from lstm import lstm_model
from feedback import feedback_form
#from sarima_model import sarima_model
from spectral_analysis import spectral_analysis 


# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "username" not in st.session_state:
    st.session_state.username = ""

# Function to set the current page
def set_page(page_name):
    st.session_state.page = page_name

# Function to log out
def logout():
    st.session_state.authenticated = False
    st.session_state.page = "Login"
    st.session_state.username = ""


# Function to display footer with copyright information
def footer():
    st.markdown(
        """
        <style>
        footer {
            visibility: hidden;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f9f9f9;
            text-align: center;
            padding: 10px;
            font-size: small;
        }
        </style>
        <div class="footer">
            <p>© 2024 Trade Bot. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Main App Logic
def main():
    if st.session_state.authenticated:
        
        # Sidebar navigation
        #st.sidebar.title(f"Welcome back, {st.session_state.username}!")

        # Automatically show the data load module after login
        if st.session_state.page == "Login":
            st.session_state.page = "Load Data"  # Redirect to Load Data immediately after login


        st.sidebar.title("Stock Bot Menu")
        st.sidebar.button("Load Stock Data", on_click=set_page, args=("Load Data",), key="load_data_button")
        st.sidebar.button("Candlestick", on_click=set_page, args=("Visualize Data",), key="visualize_data_button")
        st.sidebar.button("Moving Analysis", on_click=set_page, args=("Moving Analysis",), key="moving_analysis_button")
        st.sidebar.button("Price Returns", on_click=set_page, args=("Price Returns",), key="price_returns_button")
        st.sidebar.button("Slope of Trend Lines", on_click=set_page, args=("Trend Slope",), key="trend_slope_button")
        st.sidebar.button("Volatility Indicators", on_click=set_page, args=("Volatility Indicators",), key="volatility_button")
        st.sidebar.button("Spectral Analysis", on_click=set_page, args=("Spectral Analysis",), key="spectral_analysis_button")
        st.sidebar.button("Time Series Decomposition", on_click=set_page, args=("Decomposition",), key="decomposition_button")
        st.sidebar.button("Preprocess Data", on_click=set_page, args=("Preprocess Data",), key="preprocess_data_button")
        #st.sidebar.button("Check Stationarity", on_click=set_page, args=("Check Stationarity",), key="check_stationarity_button")
        #st.sidebar.button("Transform Data", on_click=set_page, args=("Transform Data",), key="transform_data_button")
        st.sidebar.button("ARIMA Model", on_click=set_page, args=("ARIMA Model",), key="arima_button")
        st.sidebar.button("Random Forest Model", on_click=set_page, args=("Random Forest",), key="random_forest_button")
        #st.sidebar.button("SARIMA Model", on_click=set_page, args=("SARIMA Model",), key="sarima_button")
        #st.sidebar.button("LSTM Model", on_click=set_page, args=("LSTM",), key="lstm_button")
        st.sidebar.button("Contact Us", on_click=set_page, args=("Contact Us",), key="contact_us_button")
        st.sidebar.button("Feedback", on_click=set_page, args=("Feedback",), key="feedback_button")
        st.sidebar.button("Logout", on_click=logout, key="logout_button")

        # Display the current page
        if st.session_state.page == "Load Data":
            load_data()
        elif st.session_state.page == "Visualize Data":
            visualize_data()
        elif st.session_state.page == "Moving Analysis":
            moving_indicators()
        elif st.session_state.page == "Price Returns":
            calculate_price_returns()
        elif st.session_state.page == "Trend Slope":
            compute_trend_slope()
        elif st.session_state.page == "Volatility Indicators":
            display_volatility_indicators()
        elif st.session_state.page == "Decomposition":
            decompose_time_series()
        elif st.session_state.page == "Spectral Analysis":
            spectral_analysis()
        elif st.session_state.page == "Preprocess Data":
            preprocess_data()
        #elif st.session_state.page == "Check Stationarity":
        #    check_stationarity()
        #elif st.session_state.page == "Transform Data":
        #    apply_transformations()
        elif st.session_state.page == "ARIMA Model":
            arima_model()
        elif st.session_state.page == "Random Forest":
            random_forest_model()
        #elif st.session_state.page == "LSTM":
        #    lstm_model()
        #elif st.session_state.page == "SARIMA Model":
        #    sarima_model()
        elif st.session_state.page == "Spectral Analysis":
            spectral_analysis()
        elif st.session_state.page == "Contact Us":
            contact_us()
        elif st.session_state.page == "Feedback":
            feedback_form() 
    else:
        # User options for login and registration
        st.sidebar.title("User Options")
        if st.sidebar.button("Login", key="login_button"):
            set_page("Login")
        if st.sidebar.button("Register", key="register_button"):
            set_page("Register")
        # if st.sidebar.button("Contact Us", key="contact_us_guest_button"):
        #     contact_us()
        # Render the appropriate page
        if st.session_state.page == "Login":
            login()
        elif st.session_state.page == "Register":
            register_user()

    # Footer
    footer()

if __name__ == "__main__":
    main()