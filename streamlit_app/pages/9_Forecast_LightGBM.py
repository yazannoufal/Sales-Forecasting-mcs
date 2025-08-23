# streamlit_app/pages/9_Forecast_LightGBM.py

import sys, os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# Add parent directory to path to import utility scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.data_loader import load_and_clean_data
from forecast_pipeline_lightgbm import train_and_forecast_lgbm

# Page configuration
st.set_page_config(page_title="🗓️ LightGBM Forecast", layout="wide")
st.title("🗓️ LightGBM Demand Forecast")

st.markdown("This page uses a LightGBM model to predict future demand based on your historical data. It analyzes sales patterns, seasonality, and other factors to provide a detailed forecast.")

# Load the dataset (cached for performance)
@st.cache_data
def get_data():
    return load_and_clean_data('data/sales_data.csv')

df = get_data()

if df.empty:
    st.warning("⚠️ Please ensure the 'sales_data.csv' file exists and is not empty.")
    st.stop()
    
st.markdown("---")

# User inputs
st.sidebar.header("📊 Forecast Settings")
days_to_forecast = st.sidebar.slider("Select number of days to forecast:", 7, 90, 365)
run_forecast = st.sidebar.button("Run Forecast")

# Initial data visualization
if not run_forecast:
    st.subheader("Historical Daily Demand")
    daily_demand = df.groupby('Date')['Units Ordered'].sum().reset_index()
    fig = px.line(daily_demand, x='Date', y='Units Ordered', title='Historical Daily Demand')
    fig.update_layout(xaxis_title="Date", yaxis_title="Units Ordered")
    st.plotly_chart(fig, use_container_width=True)

# Run the forecast
if run_forecast:
    st.info(f"🔮 Generating a {days_to_forecast}-day demand forecast using LightGBM...")
    
    with st.spinner('Training model and forecasting...'):
        try:
            forecast_df = train_and_forecast_lgbm(df, days_to_forecast)
            
            st.success("✅ Forecast generated successfully!")
            
            # Combine historical and forecast data for visualization
            historical_demand = df.groupby('Date')['Units Ordered'].sum().reset_index()
            historical_demand['Type'] = 'Historical'
            
            forecast_df = forecast_df.rename(columns={'Predicted Demand': 'Units Ordered'})
            forecast_df['Type'] = 'Forecast'
            
            combined_df = pd.concat([historical_demand, forecast_df])
            
            # Plotting the combined data
            fig = px.line(combined_df, x='Date', y='Units Ordered', color='Type',
                          title=f'Daily Demand: Historical vs. {days_to_forecast}-Day Forecast')
            fig.update_layout(xaxis_title="Date", yaxis_title="Units Ordered")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast data
            st.subheader("📋 Forecasted Data")
            st.write(forecast_df)
            
            # Provide download link for forecast
            csv_forecast = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast as CSV",
                data=csv_forecast,
                file_name=f'lightgbm_forecast_{date.today()}.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"An error occurred during forecasting: {e}")