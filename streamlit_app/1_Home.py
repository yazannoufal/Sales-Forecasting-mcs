# streamlit_app/1_Home.py

import sys, os
import streamlit as st
import pandas as pd
import plotly.express as px

# Add parent directory to path to import utility scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import load_and_clean_data
from utils.recommendation_engine import smart_recommendations

# Page configuration
st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("🏡 Sales & Operations Dashboard")
st.markdown("Welcome to the interactive sales forecasting and analytics system. This dashboard provides a quick overview of your business performance.")

# Load data (cached for performance)
@st.cache_data
def get_data():
    return load_and_clean_data('data/sales_data.csv')

df = get_data()

if df.empty:
    st.warning("⚠️ Please ensure the 'sales_data.csv' file exists and is not empty.")
    st.stop()

# --- 1. Key Performance Indicators (KPIs) ---
st.header("Key Performance Indicators (KPIs)")
total_sales = int(df['Units Sold'].sum())
avg_daily_demand = round(df.groupby('Date')['Units Ordered'].sum().mean(), 2)

# التعديل: استخدام 'Product ID' بدلاً من 'Product Name'
unique_products = df['Product ID'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Units Sold", f"{total_sales:,}")
col2.metric("Average Daily Demand", f"{avg_daily_demand:,}")
col3.metric("Unique Products", unique_products)

# --- 2. Interactive Visualizations ---
st.header("Quick Insights & Trends")
st.markdown("Use the slider to filter data by date range.")

# Date filter
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

if filtered_df.empty:
    st.warning("No data available for the selected date range.")
else:
    # Demand over time
    daily_demand = filtered_df.groupby('Date')['Units Ordered'].sum().reset_index()
    fig_demand = px.line(daily_demand, x='Date', y='Units Ordered', title='Daily Demand')
    st.plotly_chart(fig_demand, use_container_width=True)

    # Sales by category
    category_sales_df = filtered_df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False).reset_index()
    fig_category = px.bar(category_sales_df, x='Category', y='Units Sold', title='Total Units Sold by Category')
    st.plotly_chart(fig_category, use_container_width=True)
    
# --- 3. Quick Recommendations ---
st.header("Smart Seasonal Recommendations")
st.markdown("Get a quick recommendation based on historical seasonal trends.")

recommendation_date = st.date_input("Select a date to get recommendations", pd.to_datetime('today'))

if st.button("Generate Quick Recommendations"):
    # التعديل: استدعاء الدالة بشكل صحيح الآن
    recommendations = smart_recommendations(df, pd.to_datetime(recommendation_date))
    
    if recommendations:
        st.markdown("Based on the historical data, here are some smart recommendations:")
        for rec in recommendations:
            st.success(f"💡 {rec}")
    else:
        st.info("No specific recommendations for this season.")