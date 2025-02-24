#login.py
import streamlit as st
from database import authenticate_user, initialize_database

def login():
    """
    Login page for user authentication.
    """
    try:

        st.title("Login")
        st.write("Welcome to the Trade Bot! Please log in to continue.")

        # Initialize the database
        try:
            initialize_database()
        except Exception as db_error:
            st.error(f"Database Initialization Error: {db_error}")
            return False

        # Input fields for login credentials
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        # Login button
        if st.button("Login"):
            authenticated, role = authenticate_user(username, password)
            if authenticated:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                #st.success("Login successful! Welcome back.")
                st.rerun()  # Refresh the page to show authenticated content
                st.session_state.page = "Load Data"  # Redirect to Load Data after login
            else:
                st.error("Invalid username or password. Please try again.")


        return st.session_state.get("authenticated", False)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False
def logout():
    """
    Logs out the user by resetting the session state.
    """
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.page = "Login"
        st.session_state.username = ""
        if 'data' in st.session_state:
            del st.session_state['data']
        st.success("You have been logged out.")
        st.rerun()
