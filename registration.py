#registration.py
import streamlit as st
from database import add_user, initialize_database

def register_user():
    """
    Registration page for adding new users.
    """
    st.title("Register New User")
    st.write("Create your account to access the Stock Price Prediction Data Product.")

    # Initialize the database
    initialize_database()

    # Input fields for new user credentials
    username = st.text_input("Choose a Username", placeholder="Enter your username")
    password = st.text_input("Choose a Password", type="password", placeholder="Enter your password")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")

    # Registration button
    if st.button("Register"):
        if not username or not password or not confirm_password:
            st.error("All fields are required.")
        elif password != confirm_password:
            st.error("Passwords do not match. Please try again.")
        else:
            try:
                add_user(username, password)
                st.success("Registration successful! You can now log in.")
                #st.write("Go to the [Login Page](#)")
            except Exception as e:
                st.error(f"Error: {str(e)}")
