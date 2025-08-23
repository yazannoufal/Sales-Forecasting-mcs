import pandas as pd
from prophet import Prophet

def predict_future_prophet(days=7, return_true=False):
    """
    Build and fit Prophet model for time series forecasting.
    """
    df = pd.read_csv("data/sales_data.csv", parse_dates=['Date'])
    daily = df.groupby('Date')["Units Ordered"].sum().reset_index()
    daily = daily.rename(columns={"Date": "ds", "Units Ordered": "y"})

    model = Prophet()
    model.fit(daily)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].tail(days)
    result = result.rename(columns={"ds": "Date", "yhat": "Predicted Demand"})

    if return_true:
        y_true = daily['y'].values[-days:]
        return result, y_true

    return result
