# utils/recommendation_engine.py

import pandas as pd
from datetime import datetime
import calendar

def get_season(month: int) -> str:
    """Returns the season based on the month."""
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

def smart_recommendations(df: pd.DataFrame, current_date: datetime) -> list:
    """
    Generates smart recommendations based on historical seasonal trends.
    
    Args:
        df: The cleaned sales data DataFrame.
        current_date: The date to base the recommendations on.
    
    Returns:
        A list of recommendation strings.
    """
    if 'Date' not in df.columns:
        return ["Error: 'Date' column not found in data."]
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    # Check if 'Seasonality' column exists, if not, create it
    if 'Seasonality' not in df.columns:
        df['Seasonality'] = df['Month'].apply(get_season)

    recommendations = []
    
    current_season = get_season(current_date.month)

    # --- Rule 1: High-demand products in the current season ---
    seasonal_data = df[df['Seasonality'] == current_season]
    if not seasonal_data.empty:
        seasonal_sales = seasonal_data.groupby('Product ID')['Units Sold'].sum().sort_values(ascending=False).reset_index()
        
        if not seasonal_sales.empty:
            top_products = seasonal_sales.head(3)
            for _, row in top_products.iterrows():
                product_id = row['Product ID']
                # Try to get the category, handle potential key errors
                try:
                    product_category = seasonal_data[seasonal_data['Product ID'] == product_id]['Category'].iloc[0]
                except KeyError:
                    product_category = "N/A"
                recommendations.append(f"Product '{product_id}' (Category: {product_category}) historically performs well in the {current_season} season. Consider increasing inventory or running a targeted promotion.")

    # --- Rule 2: Products with recent sales spikes in other regions ---
    if 'Region' in df.columns:
        recent_data = df[df['Date'] >= current_date - pd.Timedelta(days=30)]
        if not recent_data.empty:
            regional_sales = recent_data.groupby(['Region', 'Product ID'])['Units Sold'].sum().reset_index()
            
            if not regional_sales.empty:
                top_regional_product = regional_sales.sort_values('Units Sold', ascending=False).iloc[0]
                recommendations.append(f"Product '{top_regional_product['Product ID']}' is showing a significant sales spike in the '{top_regional_product['Region']}' region. This could be a market opportunity for other regions.")
            
    # --- Rule 3: Products popular in the previous season ---
    previous_month = (current_date.month - 2) % 12 + 1 if current_date.month > 1 else 12
    previous_season = get_season(previous_month)
    
    previous_season_data = df[df['Seasonality'] == previous_season]
    if not previous_season_data.empty:
        sales_in_previous_season = previous_season_data.groupby('Product ID')['Units Sold'].sum().sort_values(ascending=False).head(1).reset_index()
        if not sales_in_previous_season.empty:
            product_id = sales_in_previous_season['Product ID'].iloc[0]
            recommendations.append(f"Product '{product_id}' was a top seller last season. Consider running a clearance sale to manage inventory before it becomes 'dead stock'.")
    
    if not recommendations:
        return ["No specific recommendations were found based on the provided data for this season. Please check your data or try a different date."]
    
    return recommendations