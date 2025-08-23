# streamlit_app/pages/3_SWOT_Analysis.py

import sys, os
import streamlit as st
import pandas as pd
from datetime import date, timedelta

# Add parent directory to path to import utility scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.data_loader import load_and_clean_data
from utils.swot_pipeline import generate_swot_from_data

# Page configuration
st.set_page_config(page_title="📄 SWOT Analysis", layout="wide")
st.title("📄 SWOT Analysis")

st.markdown("This page automatically generates a SWOT analysis based on your sales data. The analysis is based on the entire dataset available.")

# Load the dataset (cached for performance)
@st.cache_data
def get_data():
    return load_and_clean_data('data/sales_data.csv')

df = get_data()

# Check if DataFrame is empty
if df.empty:
    st.warning("⚠️ Please ensure the 'sales_data.csv' file exists and is not empty.")
    st.stop()

# --- Displaying Key Metrics ---
st.subheader("💡 Key Metrics for Analysis")
st.info("The analysis below uses dynamic thresholds based on these numbers to provide a more detailed output.")

daily_demand = df.groupby('Date')['Units Ordered'].sum()
avg_daily_demand = round(daily_demand.mean(), 2)
median_daily_demand = round(daily_demand.median(), 2)

product_sales = df.groupby('Product ID')['Units Sold'].sum()
avg_product_sales = round(product_sales.mean(), 2)
median_product_sales = round(product_sales.median(), 2)
total_unique_products = df['Product ID'].nunique()

col1, col2 = st.columns(2)
col1.metric("Average Daily Demand", f"{avg_daily_demand:,}")
col2.metric("Median Daily Demand", f"{median_daily_demand:,}")

col3, col4 = st.columns(2)
col3.metric("Average Product Sales", f"{avg_product_sales:,}")
col4.metric("Median Product Sales", f"{median_product_sales:,}")

st.metric("Total Unique Products", total_unique_products)

st.markdown("---")

# Run analysis button
if st.button("Generate SWOT Analysis"):
    st.info("Generating analysis based on the full dataset...")
    
    # We now pass the same DataFrame to the pipeline, as it contains all the necessary data
    swot_result = generate_swot_from_data(df)
    
    st.subheader("Your SWOT Analysis")
    st.markdown("---")
    
    st.markdown("### 💪 Strengths")
    for strength in swot_result['Strengths']:
        st.success(strength)

    st.markdown("---")

    st.markdown("### 📉 Weaknesses")
    for weakness in swot_result['Weaknesses']:
        st.error(weakness)

    st.markdown("---")

    st.markdown("### 🚀 Opportunities")
    for opportunity in swot_result['Opportunities']:
        st.info(opportunity)
        
    st.markdown("---")

    st.markdown("### ⚠️ Threats")
    for threat in swot_result['Threats']:
        st.warning(threat)