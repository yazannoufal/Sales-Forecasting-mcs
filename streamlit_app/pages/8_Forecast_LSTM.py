# streamlit_app/pages/📈_Forecast(LSTM).py

import sys, os
import streamlit as st
import pandas as pd
import plotly.express as px

# Add parent directory to path to import utility scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from forecast_pipeline_lstm import predict_future_lstm

# Page configuration
st.set_page_config(page_title="📈 Forecast (LSTM)", layout="wide")
st.title("📈 Forecast with LSTM Model")

st.markdown("This page uses a pre-trained **Long Short-Term Memory (LSTM)** neural network model to forecast future demand.")

# User input for number of days to forecast
days = st.slider("Select number of days to forecast", 3, 365, 7)

# Run forecast button
if st.button("Run LSTM Forecast"):
    with st.spinner("Generating forecast..."):
        # Run the forecast pipeline
        forecast_df = predict_future_lstm(days)
        
        # Display the forecast data
        st.subheader("Forecasted Data")
        st.dataframe(forecast_df)
        
        # Plot the forecast
        st.subheader("Forecast Visualization")
        fig = px.line(forecast_df, x='Date', y='Predicted Demand', title="LSTM Forecast")
        st.plotly_chart(fig, use_container_width=True)