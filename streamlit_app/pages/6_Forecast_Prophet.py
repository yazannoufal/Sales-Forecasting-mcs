import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from forecast_pipeline_prophet import predict_future_prophet
import plotly.express as px

st.title("📈 Forecast with Prophet")

days = st.slider("Select number of days to forecast", 3, 365, 7)
if st.button("Run Forecast"):
    forecast_df = predict_future_prophet(days)
    st.dataframe(forecast_df)

    fig = px.line(forecast_df, x='Date', y='Predicted Demand', title="Prophet Forecast")
    st.plotly_chart(fig, use_container_width=True)
