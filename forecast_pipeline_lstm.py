import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

def predict_future_lstm(days=7, return_true=False):
    """
    Load pre-trained LSTM model and scaler to forecast demand.
    """
    scaler = joblib.load("models/lstm_scaler.pkl")
    model = load_model("models/lstm_model.keras")

    df = pd.read_csv("data/sales_data.csv", parse_dates=["Date"])
    df_daily = df.groupby("Date")["Units Ordered"].sum().reset_index()
    data = df_daily["Units Ordered"].values.reshape(-1, 1)
    scaled = scaler.transform(data)

    X_input = scaled[-10:].reshape(1, 10, 1)
    predictions = []
    last_date = df_daily["Date"].max()

    for _ in range(days):
        pred = model.predict(X_input, verbose=0)[0][0]
        predictions.append(pred)
        X_input = np.append(X_input[:, 1:, :], [[[pred]]], axis=1)

    preds_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)

    if return_true:
        y_true = data[-days:].flatten()
        return pd.DataFrame({"Date": forecast_dates, "Predicted Demand": preds_rescaled}), y_true

    return pd.DataFrame({"Date": forecast_dates, "Predicted Demand": preds_rescaled})
