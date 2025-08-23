import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from forecast_pipeline import predict_future_arima
import plotly.express as px

st.title("📈 Forecast with ARIMA")

days = st.slider("Select number of days to forecast", 3, 30, 365)
if st.button("Run Forecast"):
    forecast_df = predict_future_arima(days)
    st.dataframe(forecast_df)

    fig = px.line(forecast_df, x='Date', y='Predicted Demand', title="ARIMA Forecast")
    st.plotly_chart(fig, use_container_width=True)
