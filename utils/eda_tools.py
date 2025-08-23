# utils/eda_tools.py

import pandas as pd
import plotly.express as px

def plot_interactive_demand(df: pd.DataFrame):
    """
    Creates an interactive line chart for total units ordered over time.
    """
    daily_demand = df.groupby('Date')['Units Ordered'].sum().reset_index()
    fig = px.line(daily_demand, x='Date', y='Units Ordered', title='Total Units Ordered Over Time')
    return fig

def category_sales(df: pd.DataFrame):
    """
    Calculates total units sold by category.
    """
    category_data = df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False)
    return category_data

def season_demand_plot(df: pd.DataFrame):
    """
    Creates a bar chart for average demand per season using Plotly.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Map months to seasons
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
        11: 'Fall', 12: 'Winter'
    }
    df['Season'] = df['Date'].dt.month.map(season_map)
    
    # Calculate average units ordered per season
    seasonal_demand = df.groupby('Season')['Units Ordered'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall']).reset_index()
    
    fig = px.bar(
        seasonal_demand,
        x='Season',
        y='Units Ordered',
        title='Average Demand by Season'
    )
    return fig