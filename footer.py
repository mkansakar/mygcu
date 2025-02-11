#footer.py
import streamlit as st

def display_footer():
    """
    Display the footer with clickable links for Feedback and Contact Us,
    along with plain text Copyright info.
    """
    st.markdown("---")  # Horizontal line separator

    footer_html = """
    <div style='text-align: center; font-size: 14px; margin-top: 30px;'>
        <a href="#feedback" style='text-decoration: none; margin-right: 20px;'>Feedback</a> |
        <a href="#contact-us" style='text-decoration: none; margin-left: 20px;'>Contact Us</a><br>
        © 2024 Stock Price Prediction Data Product | All Rights Reserved
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
