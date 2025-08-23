import joblib
import pandas as pd
from datetime import timedelta

def predict_future_arima(days=7, return_true=False):
    """
    Load pre-trained ARIMA model and generate future forecast.
    """
    model = joblib.load("models/arima_model.pkl")
    forecast = model.forecast(steps=days)
    last_date = model.data.dates[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Demand": forecast})

    if return_true:
        y_true = model.data.endog[-days:]
        return forecast_df, y_true

    return forecast_df
