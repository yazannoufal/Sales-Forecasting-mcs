# utils/data_loader.py

import pandas as pd
import streamlit as st

def load_and_clean_data(file_path):
    """
    Loads sales data from a CSV file, cleans it, and prepares it for analysis.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()

    # Define standard column names and their provided variations
    column_mapping = {
        'Date': 'Date',
        'Store ID': 'Store ID',
        'Product ID': 'Product ID',
        'Category': 'Category',
        'Region': 'Region',
        'Inventory Level': 'Inventory Level',
        'Units Sold': 'Units Sold',
        'Units Ordered': 'Units Ordered',
        'Demand Forecast': 'Demand Forecast',
        'Price': 'Price',
        'Discount': 'Discount',
        'Weather Condition': 'Weather Condition',
        'Holiday/Promotion': 'Holiday/Promotion',
        'Competitor Pricing': 'Competitor Pricing',
        'Seasonality': 'Seasonality'
    }

    # Standardize column names
    df.rename(columns=column_mapping, inplace=True)

    # Data cleaning and preparation
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values('Date', inplace=True)
    
    df.columns = df.columns.str.strip()
    
    # Fill missing values for numerical columns
    numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert 'Holiday/Promotion' to boolean
    if 'Holiday/Promotion' in df.columns:
        df['Holiday/Promotion'] = df['Holiday/Promotion'].astype(bool)

    # Forward-fill any remaining missing data
    df.fillna(method='ffill', inplace=True)

    return df