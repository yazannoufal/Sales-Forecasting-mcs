from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_metrics(true, predicted):
    """
    Calculate MAE and RMSE between true and predicted values.
    """
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, rmse
