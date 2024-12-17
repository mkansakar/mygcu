import streamlit as st

def contact_us():
    """
    Display a 'Contact Us' page for users to send feedback or inquiries.
    """
    st.title("Contact Us")
    st.subheader("We'd love to hear from you!")

    # Contact Information
    st.write("""
    For any inquiries, suggestions, or support, please feel free to reach out to us:
    
    - **Email**: mkansakar2013[@]gmail.com  
    - **Phone**: +1-781-609-8099  
    - **Address**: 5 Warren St. Arlington, MA 02474
    """)

    # Contact Form
    st.subheader("Send Us a Message")
    with st.form("contact_form"):
        name = st.text_input("Your Name", placeholder="Enter your full name")
        email = st.text_input("Your Email", placeholder="Enter your email address")
        message = st.text_area("Your Message", placeholder="Write your message here...")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if name and email and message:
                st.success("Thank you for reaching out! We'll get back to you shortly.")
            else:
                st.error("Please fill out all fields before submitting.")
