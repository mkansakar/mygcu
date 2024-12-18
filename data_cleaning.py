import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data():
    """
    Data Cleaning and Preprocessing Module with Missing Value Check.
    """
    if 'data' in st.session_state:
        st.title("Data Cleaning and Preprocessing")

        # Load the dataset
        data = st.session_state['data'].copy()

        # Check for missing values
        st.subheader("Check for Missing Values")
        if data.isnull().values.any():
            st.warning(f"Missing values detected! {data.isnull().sum().sum()} total missing values found.")
            
            # Handle missing values
            missing_method = st.radio(
                "Select how to handle missing values:",
                options=["Forward Fill", "Backward Fill", "Mean Replacement", "Remove Rows"],
                key="missing_method"
            )

            if st.button("Apply Missing Value Handling"):
                if missing_method == "Forward Fill":
                    data.fillna(method="ffill", inplace=True)
                elif missing_method == "Backward Fill":
                    data.fillna(method="bfill", inplace=True)
                elif missing_method == "Mean Replacement":
                    data.fillna(data.mean(), inplace=True)
                elif missing_method == "Remove Rows":
                    data.dropna(inplace=True)
                st.success("Missing values handled successfully!")
                st.dataframe(data.head())
        else:
            st.success("No missing values found in the dataset!")

        # Handle duplicates
        st.subheader("Check for Duplicates")
        if data.duplicated().any():
            if st.checkbox("Remove Duplicate Rows", key="remove_duplicates"):
                before_rows = data.shape[0]
                data.drop_duplicates(inplace=True)
                after_rows = data.shape[0]
                st.write(f"Removed {before_rows - after_rows} duplicate rows.")
                st.dataframe(data.head())
        else:
            st.success("No duplicate rows found!")

        # Normalize/Scale data
        st.subheader("Normalize Data")
        if st.checkbox("Apply Normalization (Min-Max Scaling)", key="normalize"):
            scaler = MinMaxScaler()
            columns_to_scale = ["Open", "High", "Low", "Close", "Volume"]
            data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
            st.success("Data normalized successfully!")
            st.dataframe(data.head())

        # Save processed data to session state
        if st.button("Save Cleaned Data", key="save_cleaned_data"):
            st.session_state['cleaned_data'] = data
            st.success("Cleaned data saved to session!")
    else:
        st.error("Please load the data first from the sidebar on the left.")
