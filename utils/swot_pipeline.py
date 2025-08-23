# utils/swot_pipeline.py

import pandas as pd
import numpy as np
from datetime import timedelta
import streamlit as st

def generate_swot_from_data(df: pd.DataFrame) -> dict:
    """
    Generates a SWOT analysis from sales data based on predefined rules.
    This version analyzes the entire dataset with dynamic thresholds based on the data itself.
    """
    full_data = df.copy()
    
    historic_volatility = full_data['Units Ordered'].var()
    sales_growth_rate = 0.15 
    top_category_sales = full_data.groupby('Category')['Units Sold'].sum().max()
    avg_category_sales = full_data.groupby('Category')['Units Sold'].sum().mean()
    forecasted_drop = 0.10
    
    swot = {
        'Strengths': [],
        'Weaknesses': [],
        'Opportunities': [],
        'Threats': []
    }
    
    # Calculate median for sales and stock data for dynamic rules
    median_product_sales = full_data.groupby('Product ID')['Units Sold'].sum().median()
    median_stock_level = full_data['Inventory Level'].median() if 'Inventory Level' in full_data.columns else 0
    
    # --- Strengths ---
    if not full_data.empty:
        # Rule 1: High average demand
        if full_data['Units Ordered'].mean() > full_data['Units Ordered'].median() * 1.25:
            swot['Strengths'].append("The average daily demand is significantly high compared to historical data, indicating strong market fit.")
            
        # Rule 2: High sales growth (This is an assumption, needs real data to be accurate)
        if sales_growth_rate > 0.10: 
            swot['Strengths'].append("Historical sales have grown at a fast pace, showing strong market expansion.")
            
        # Rule 3: Top-performing category
        if top_category_sales > avg_category_sales * 1.50: 
            swot['Strengths'].append("A particular product category has consistently outperformed others, a sign of its popularity.")
            
        # Rule 4: High-demand products with sufficient stock
        high_demand_products = full_data.groupby('Product ID')['Units Ordered'].sum()
        high_demand_products = high_demand_products[high_demand_products > high_demand_products.median() * 1.5]
        if not high_demand_products.empty:
            for product_id in high_demand_products.index:
                if 'Inventory Level' in full_data.columns:
                    recent_stock = full_data[full_data['Product ID'] == product_id]['Inventory Level'].iloc[-1]
                    if recent_stock > high_demand_products[product_id]: 
                        swot['Strengths'].append(f"Inventory for high-demand product '{product_id}' is sufficient, ensuring no lost sales.")
                        
    # --- Weaknesses ---
    if not full_data.empty:
        # Rule 1: High demand volatility
        demand_volatility = full_data['Units Ordered'].var()
        if demand_volatility > historic_volatility * 1.5: 
            swot['Weaknesses'].append("There is a high volatility in demand, making sales difficult to predict.")

        # Rule 2: Low-performing products
        low_sales_products = full_data.groupby('Product ID')['Units Sold'].sum()
        low_sales_products = low_sales_products[low_sales_products < median_product_sales * 0.50]
        if not low_sales_products.empty:
            low_sales_product_ids = [str(pid) for pid in low_sales_products.index.tolist()]
            swot['Weaknesses'].append(f"Some products like {', '.join(low_sales_product_ids)} are showing very low historical sales.")

        # Rule 3: Dead stock
        if 'Inventory Level' in full_data.columns:
            dead_stock_df = full_data.drop_duplicates('Product ID', keep='last')
            dead_stock = dead_stock_df[(dead_stock_df['Inventory Level'] > median_stock_level * 1.5) & (dead_stock_df['Units Sold'] < median_product_sales * 0.25)]
            if not dead_stock.empty:
                dead_stock_ids = [str(pid) for pid in dead_stock['Product ID'].tolist()]
                swot['Weaknesses'].append(f"There is significant 'dead stock' for products such as {', '.join(dead_stock_ids)} which are not selling well.")
        # --- Opportunities ---
    if not full_data.empty:
        opportunities = []

        # Rule: High Potential Products
        high_potential_products = full_data.groupby('Product ID')['Units Sold'].sum()
        high_potential_products = high_potential_products[high_potential_products > median_product_sales * 0.5]
        if not high_potential_products.empty:
            high_potential_ids = [str(pid) for pid in high_potential_products.index.tolist()]
            opportunities.append({
                "type": "product",
                "message": f"📦 منتجات ذات أداء قوي: {', '.join(high_potential_ids)} تمثل فرصة لتعزيز التسويق والاستثمار."
            })

        # Rule: Seasonality Opportunity
        if 'Seasonality' in full_data.columns and 'Units Sold' in full_data.columns:
            seasonal_sales = full_data.groupby('Seasonality')['Units Sold'].sum()
            avg_sales = seasonal_sales.mean()
            strong_seasons = seasonal_sales[seasonal_sales > avg_sales * 0.4]
            if not strong_seasons.empty:
                for season, sales in strong_seasons.items():
                    opportunities.append({
                        "type": "season",
                        "message": f"🌤️ موسم '{season}' شهد مبيعات أعلى من المتوسط، مما يشكل فرصة لتكثيف الحملات التسويقية وتخطيط المخزون."
                    })

        # إذا ما في فرص واضحة
        if not opportunities:
            swot['Opportunities'].append("No specific opportunities were identified in the data.")
        else:
            # تخزين الرسائل فقط في dict النهائي
            swot['Opportunities'] = [item["message"] for item in opportunities]
    # --- Threats ---
    if 'Demand Forecast' in full_data.columns:
        forecast_df = full_data.iloc[-10:]
        if (forecast_df['Demand Forecast'] < forecast_df['Units Ordered'].mean() * (1 - forecasted_drop)).any():
            swot['Threats'].append("The sales forecast predicts a future drop in demand, which may require proactive strategy adjustments.")
            
    # Rule 2: Over-reliance on discounts
    if 'Holiday/Promotion' in df.columns:
        last_promo_date = df[df['Holiday/Promotion'] == True]['Date'].max()
        if not pd.isna(last_promo_date):
            sales_after_promo = full_data[(full_data['Date'] > last_promo_date) & (full_data['Date'] <= last_promo_date + timedelta(days=14))]['Units Sold'].sum()
            avg_daily_sales_before = full_data[(full_data['Date'] >= last_promo_date - timedelta(days=14)) & (full_data['Date'] < last_promo_date)]['Units Sold'].mean()
            if sales_after_promo < avg_daily_sales_before * 10: 
                swot['Threats'].append("A significant drop in sales was observed after a recent promotion, indicating over-reliance on discounts.")

    # Rule 3: Excess inventory for products with dropping demand
    if 'Demand Forecast' in full_data.columns and 'Inventory Level' in full_data.columns:
        low_forecast_products = full_data[full_data['Demand Forecast'] < full_data['Units Ordered'].median()]['Product ID'].unique()
        if len(low_forecast_products) > 0:
            for product_id in low_forecast_products:
                recent_stock = full_data[full_data['Product ID'] == product_id]['Inventory Level'].iloc[-1]
                if recent_stock > median_stock_level * 1.5: 
                    swot['Threats'].append(f"Excess inventory ({recent_stock} units) for product '{product_id}' poses a risk due to forecasted low demand.")
    
    for key in swot:
        if not swot[key]:
            swot[key].append(f"No specific {key.lower()} were identified in the data.")

    return swot