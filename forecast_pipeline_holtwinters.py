import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

def predict_future_holtwinters(days=7, return_true=False):
    """
    Use Holt-Winters method to forecast future demand.
    """
    df = pd.read_csv("data/sales_data.csv", parse_dates=['Date'])
    daily = df.groupby('Date')['Units Ordered'].sum().reset_index()
    daily.set_index('Date', inplace=True)

    model = ExponentialSmoothing(
        daily['Units Ordered'],
        trend='add',
        seasonal='add',
        seasonal_periods=7
    ).fit()

    forecast = model.forecast(days)
    forecast_dates = pd.date_range(start=daily.index[-1] + timedelta(days=1), periods=days)

    df_out = pd.DataFrame({"Date": forecast_dates, "Predicted Demand": forecast.values})

    if return_true:
        y_true = daily['Units Ordered'].values[-days:]
        return df_out, y_true

    return df_out
