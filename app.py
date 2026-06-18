import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import datetime
from data import fetch_stock_data, add_features

# Load the saved model
model = joblib.load("model.pkl")

# App title
st.title("📈 Stock Market Prediction Dashboard")

# User input
stock_name = st.text_input("Enter Stock Ticker", value="^NSEI")

if stock_name:
    # Fetch and prepare data
    data = fetch_stock_data(stock_name)
    data = add_features(data)

    # Features
    features = ["MA_20", "MA_50", "Yesterday", "Last_week", "RSI"]

    # Predict tomorrow's open using today's last row
    last_row = data[features].iloc[-1].values.reshape(1, -1)
    tomorrow_open = model.predict(last_row)[0]

    # Tomorrow's date
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    # Show tomorrow's predicted open price
    st.metric(
        label=f"Predicted Open for {tomorrow}",
        value=f"₹{tomorrow_open:.2f}"
    )

    # Plot actual open prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Open"].squeeze(),
        name="Actual Open",
        line=dict(color="blue")
    ))
    fig.update_layout(
        title=f"{stock_name} - Open Price History",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig)

    # Show recent raw data
    st.subheader("Recent Data")
    st.dataframe(data.tail(10))