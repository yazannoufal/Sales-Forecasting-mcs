# streamlit_app/pages/2_Detailed_Dashboard.py

import sys, os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import utility scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.data_loader import load_and_clean_data
from utils.eda_tools import plot_interactive_demand, category_sales, season_demand_plot

# Page configuration
st.set_page_config(page_title="📊 Sales Dashboard", layout="wide")
st.title("📊 Detailed Sales & Analytics Dashboard")
st.markdown("Dive deeper into the data with detailed visualizations and trend analysis.")

# Load the dataset (cached for performance)
@st.cache_data
def get_data():
    return load_and_clean_data('data/sales_data.csv')

df = get_data()

# Check if DataFrame is empty
if df.empty:
    st.warning("⚠️ Please ensure the 'sales_data.csv' file exists and is not empty.")
    st.stop()

# --- 1. Total Units Ordered Over Time ---
st.header("📈 Total Units Ordered Over Time")
fig_demand = plot_interactive_demand(df)
st.plotly_chart(fig_demand, use_container_width=True)

# --- 2. Sales Distribution by Category & Region ---
st.header("🧺 Sales by Category & Region")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Units Sold by Category")
    category_data = category_sales(df)
    st.bar_chart(category_data)

with col2:
    st.subheader("Units Sold by Region")
    region_data = df.groupby('Region')['Units Sold'].sum().sort_values(ascending=False)
    st.bar_chart(region_data)

# --- 3. Seasonal Demand Analysis ---
st.header("🍂 Seasonal Demand Analysis")
fig_season = season_demand_plot(df)
st.plotly_chart(fig_season, use_container_width=True)

# --- 4. Correlation with Promotions ---
if 'Holiday/Promotion' in df.columns:
    st.header("📅 Impact of Holidays & Promotions")
    promo_df = df.groupby(['Date', 'Holiday/Promotion'])['Units Ordered'].sum().reset_index()
    fig_promo = go.Figure()
    fig_promo.add_trace(go.Scatter(x=promo_df['Date'], y=promo_df['Units Ordered'],
                                   mode='lines+markers', name='Units Ordered',
                                   marker=dict(color='skyblue')))
    
    promotions = promo_df[promo_df['Holiday/Promotion'] == True]
    if not promotions.empty:
        fig_promo.add_trace(go.Scatter(x=promotions['Date'], y=promotions['Units Ordered'],
                                       mode='markers', name='Holiday/Promotion',
                                       marker=dict(color='red', size=10)))
    
    fig_promo.update_layout(title="Daily Demand with Promotions")
    st.plotly_chart(fig_promo, use_container_width=True)