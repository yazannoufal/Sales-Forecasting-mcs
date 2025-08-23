# utils/forecast_pipeline_lightgbm.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta

def create_features(df):
    """
    Creates time series features from the date column.
    """
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Simple lag features
    for lag in range(1, 4):
        df[f'lag_Units Ordered_{lag}'] = df['Units Ordered'].shift(lag)
    
    return df

def train_and_forecast_lgbm(df, days_to_forecast):
    """
    Trains a LightGBM model and generates a forecast for the specified number of days.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.set_index('Date', inplace=True)
    
    # Create features
    df = create_features(df.reset_index()).set_index('Date')
    df.dropna(inplace=True)
    
    # Define features and target based on the provided column list
    irrelevant_cols = ['Units Ordered', 'Units Sold', 'Demand Forecast', 'Price', 'Discount', 'Inventory Level', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality']
    
    features = [col for col in df.columns if col not in irrelevant_cols and col not in ['Date']]
    target = 'Units Ordered'
    
    # Check if a target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the data.")
        
    X = df[features]
    y = df[target]
    
    # Initialize and train the LightGBM model
    model = lgb.LGBMRegressor(objective='regression', metric='rmse')
    model.fit(X, y)
    
    # Generate the future forecast
    last_row = df.iloc[-1].copy()
    last_date = last_row.name
    
    forecast_list = []
    
    # Initialize lags with the last values from the historical data
    current_lags = {
        'lag_Units Ordered_1': last_row['Units Ordered'],
        'lag_Units Ordered_2': last_row['lag_Units Ordered_1'],
        'lag_Units Ordered_3': last_row['lag_Units Ordered_2']
    }
    
    current_date = last_date
    
    for _ in range(days_to_forecast):
        current_date += timedelta(days=1)
        
        # Create a new row for the date to forecast
        future_row_dict = {
            'Date': current_date,
            'dayofweek': current_date.dayofweek,
            'quarter': current_date.quarter,
            'month': current_date.month,
            'year': current_date.year,
            'dayofyear': current_date.dayofyear,
            'dayofmonth': current_date.day,
            'weekofyear': current_date.isocalendar().week
        }
        future_row_dict.update(current_lags)
        future_row = pd.DataFrame([future_row_dict])
        
        # Make sure the feature order is correct for the model
        future_row = future_row[features]
        
        # Predict the demand for the next day
        prediction = model.predict(future_row)[0]
        
        # Update the lags for the next iteration
        current_lags['lag_Units Ordered_3'] = current_lags['lag_Units Ordered_2']
        current_lags['lag_Units Ordered_2'] = current_lags['lag_Units Ordered_1']
        current_lags['lag_Units Ordered_1'] = prediction
        
        forecast_list.append({
            'Date': current_date,
            'Predicted Demand': max(0, prediction) # Ensure no negative predictions
        })
        
    return pd.DataFrame(forecast_list)