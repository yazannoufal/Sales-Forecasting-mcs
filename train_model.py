import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

st.set_page_config(page_title="📚 Train Models", layout="wide")
st.title("📚 Train Forecasting Models")

# Load the data
data_path = "data/sales_data.csv"
if not os.path.exists(data_path):
    st.error("❌ sales_data.csv not found in /data folder.")
    st.stop()

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values("Date")

daily_demand = df.groupby("Date")["Units Ordered"].sum()

if st.button("🚀 Train Models"):
    with st.spinner("Training ARIMA model..."):
        arima_model = ARIMA(daily_demand, order=(5, 1, 2))
        arima_fitted = arima_model.fit()
        joblib.dump(arima_fitted, "models/arima_model.pkl")
        st.success("✅ ARIMA model saved.")

    with st.spinner("Training LSTM model..."):
        series = daily_demand.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_series = scaler.fit_transform(series)

        X_lstm, y_lstm = [], []
        window = 10
        for i in range(len(scaled_series) - window):
            X_lstm.append(scaled_series[i:i+window])
            y_lstm.append(scaled_series[i+window])

        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        model_lstm = Sequential([
            LSTM(64, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_lstm, y_lstm, epochs=30, batch_size=16,
                       callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        model_lstm.save("models/lstm_model.keras")
        joblib.dump(scaler, "models/lstm_scaler.pkl")
        st.success("✅ LSTM model & scaler saved.")

    with st.spinner("Training LightGBM model..."):
        cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
        df_encoded = pd.get_dummies(df, columns=cat_cols)
        features = [col for col in df_encoded.columns if col not in ["Date", "Units Ordered"]]
        X = df_encoded[features]
        y = df_encoded["Units Ordered"]
        X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)

        model_lgbm = lgb.LGBMRegressor()
        model_lgbm.fit(X_train, y_train)
        joblib.dump(model_lgbm, "models/lightgbm_model.pkl")
        st.success("✅ LightGBM model saved.")

    st.balloons()
    st.success("🎉 All models trained successfully!")

st.info("ℹ️ Prophet and Holt-Winters models do not require training in advance. They are generated dynamically when forecasting.")
