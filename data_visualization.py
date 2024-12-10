import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def visualize_data():
    """
    Visualize stock data and display basic statistics.
    """
    if 'data' not in st.session_state:
        st.error("No data loaded. Please load stock data first.")
        return

    data = st.session_state['data']

    st.title("Data Visualization and Basic Statistics")
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Select visualization type
    st.subheader("Visualizations")
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Line Chart", "Candlestick Chart", "Volume Bar Chart"]
    )

    if chart_type == "Line Chart":
        selected_column = st.selectbox("Select Column to Plot", data.columns, index=data.columns.get_loc("Close"))
        fig = px.line(data, x=data.index, y=selected_column, title=f"{selected_column} Over Time")
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title=selected_column)
        st.plotly_chart(fig)

    elif chart_type == "Candlestick Chart":
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close']
                    )
                ]
            )
            fig.update_layout(
                title="Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )
            st.plotly_chart(fig)
        else:
            st.error("The dataset does not contain required columns for a Candlestick Chart.")

    elif chart_type == "Volume Bar Chart":
        if "Volume" in data.columns:
            fig = px.bar(data, x=data.index, y="Volume", title="Trading Volume Over Time")
            fig.update_xaxes(title="Date")
            fig.update_yaxes(title="Volume")
            st.plotly_chart(fig)
        else:
            st.error("The dataset does not contain a Volume column.")
