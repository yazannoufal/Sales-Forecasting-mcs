import os
import streamlit as st
from datetime import datetime
from pathlib import Path
import subprocess

st.set_page_config(page_title="📦 Model Status", layout="wide")
st.title("📦 Model Status Dashboard")

st.markdown("Check the current training status of your forecasting models.")

# Define model paths
models_info = {
    "ARIMA": "models/arima_model.pkl",
    "LSTM": "models/lstm_model.keras",
    "LSTM Scaler": "models/lstm_scaler.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

# Display status of each model
cols = st.columns(len(models_info))
for i, (model_name, model_path) in enumerate(models_info.items()):
    with cols[i]:
        if os.path.exists(model_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            st.success(f"✅ {model_name}")
            st.caption(f"Last trained: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error(f"❌ {model_name}")
            st.caption("Model not found.")

# Retrain models (optional button)
st.divider()
st.subheader("🔁 Retrain All Models")

if st.button("🚀 Run Training Script"):
    with st.spinner("Training in progress..."):
        try:
            subprocess.run(["python", "train_model.py"], check=True)
            st.success("✅ All models retrained successfully.")
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
