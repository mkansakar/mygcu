#feedback.py
import streamlit as st
import pandas as pd
import os

FEEDBACK_FILE = "feedback.csv"

# Function to save feedback to a CSV file
def save_feedback_to_csv(username, rating, comments):
    """
    Saves feedback to a CSV file. If the file does not exist, it creates one.
    """
    try:
        feedback_data = {
            "Name": [username],
            "Rating": [rating],
            "Comments": [comments]
        }

        feedback_df = pd.DataFrame(feedback_data)

        # Check if the file already exists
        if not os.path.isfile(FEEDBACK_FILE):
            # Create the file and add the header
            feedback_df.to_csv(FEEDBACK_FILE, index=False, mode='w')
            st.success("Feedback saved successfully!")
        else:
            # Append the feedback to the existing file without headers
            feedback_df.to_csv(FEEDBACK_FILE, index=False, mode='a', header=False)
            st.success("Thank you! Your feedback has been saved successfully.")
    except Exception as e:
        st.error(f"An error occurred while saving your feedback: {e}")

# Feedback form
def feedback_form():
    """
    Display the feedback form for user rating and comments.
    """
    try:

        st.title("User Feedback Form")
        st.write("We value your feedback. Please rate your experience and provide suggestions.")

        # Feedback form fields
        username = st.text_input("Your Name:", placeholder="Enter your name")
        rating = st.radio("Rate your experience (1 - Poor, 5 - Excellent):", 
                        options=[1, 2, 3, 4, 5], index=4, horizontal=True)
        comments = st.text_area("Additional Comments:", placeholder="Share your thoughts...")

        if st.button("Submit Feedback"):
            if username and comments:
                save_feedback_to_csv(username, rating, comments)
            else:
                st.error("Please enter your name and comments to submit feedback.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Main function to display the feedback page

# Contact Information
    st.write("""
    About us:
    - **Email**: mkansakar2013[@]gmail.com  
    - **Phone**: +1-781-609-8099  
    - **Address**: 5 Warren St. Arlington, MA 02474
    """)