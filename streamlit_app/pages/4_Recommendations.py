import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from utils.data_loader import load_and_clean_data
from utils.recommendation_engine import smart_recommendations

df = load_and_clean_data('data/sales_data.csv')

st.title("📋 Smart Seasonal Recommendations")

if not df.empty:
    selected_date = st.date_input("Choose a date:", df['Date'].min())

    if st.button("Generate Recommendations"):
        recommendations = smart_recommendations(df, pd.to_datetime(selected_date))
        for rec in recommendations:
            st.success(rec)
else:
    st.warning("⚠️ Please ensure your CSV file is not empty.")
