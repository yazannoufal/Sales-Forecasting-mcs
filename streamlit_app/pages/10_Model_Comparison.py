import sys, os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# إضافة المسار للحصول على الدوال المساعدة
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from forecast_pipeline import predict_future_arima
from forecast_pipeline_lstm import predict_future_lstm
from forecast_pipeline_holtwinters import predict_future_holtwinters
from forecast_pipeline_prophet import predict_future_prophet
from forecast_pipeline_lightgbm import train_and_forecast_lgbm
from utils.metrics import calculate_metrics
from utils.data_loader import load_and_clean_data

# --- بداية التطبيق ---
st.title("📉 Model Comparison")

@st.cache_data
def get_data():
    return load_and_clean_data('data/sales_data.csv')
    
df = get_data()

days = st.slider("Select number of forecast days", 7, 365, 30)

if st.button("Compare Models"):
    if df.empty:
        st.warning("⚠️ Data is not available. Please check the data file.")
    else:
        with st.spinner("Running forecasts for all models..."):
            arima_df, arima_true = predict_future_arima(days, return_true=True)
            lstm_df, lstm_true = predict_future_lstm(days, return_true=True)
            prophet_df, prophet_true = predict_future_prophet(days, return_true=True)
            holt_df, holt_true = predict_future_holtwinters(days, return_true=True)
            
            lgbm_df = train_and_forecast_lgbm(df, days)
            lgbm_true = df['Demand Forecast'].iloc[-days:]

        arima_mae, arima_rmse = calculate_metrics(arima_true, arima_df["Predicted Demand"])
        lstm_mae, lstm_rmse = calculate_metrics(lstm_true, lstm_df["Predicted Demand"])
        prophet_mae, prophet_rmse = calculate_metrics(prophet_true, prophet_df["Predicted Demand"])
        holt_mae, holt_rmse = calculate_metrics(holt_true, holt_df["Predicted Demand"])
        lgbm_mae, lgbm_rmse = calculate_metrics(lgbm_true, lgbm_df["Predicted Demand"])

        st.subheader("📊 Model Error Comparison (MAE & RMSE)")
        comp_df = pd.DataFrame({
            "Model": ["ARIMA", "Prophet", "Holt-Winters", "LSTM", "LightGBM"],
            "MAE": [arima_mae, prophet_mae, holt_mae, lstm_mae, lgbm_mae],
            "RMSE": [arima_rmse, prophet_rmse, holt_rmse, lstm_rmse, lgbm_rmse]
        })
        st.dataframe(comp_df.set_index('Model'))
        
        st.info("The model with the lowest MAE and RMSE is the most accurate.")