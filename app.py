# app.py
import streamlit as st
from login import login
from registration import register_user
from data_load import load_data

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

# Main App Logic
if st.session_state.authenticated:
    # Sidebar navigation
    st.sidebar.title(f"Welcome back, {st.session_state.username}!")
    st.sidebar.button("Logout", on_click=logout, key="logout_button")
    st.sidebar.title("Stock Prediction Menu")
    st.sidebar.button("Load Stock Data", on_click=set_page, args=("Load Data",), key="load_data_button")

    # Display the current page
    if st.session_state.page == "Load Data":
        load_data()
else:
    # User options for login and registration
    st.sidebar.title("User Options")
    if st.sidebar.button("Login", key="login_button"):
        set_page("Login")
    if st.sidebar.button("Register", key="register_button"):
        set_page("Register")

    # Render the appropriate page
    if st.session_state.page == "Login":
        login()
    elif st.session_state.page == "Register":
        register_user()
