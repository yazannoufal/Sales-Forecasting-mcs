# 📈 Sales Forecasting System

An interactive forecasting and analytics system built with Python & Streamlit, using historical sales data to predict demand and generate strategic recommendations.

---

## 🔧 Features

✅ Time-series forecasting with:
- ARIMA
- LSTM
- LightGBM
- Prophet (Meta)
- Holt-Winters

✅ Smart seasonal recommendations  
✅ Interactive dashboards and visualizations  
✅ Dynamic SWOT analysis  
✅ Streamlit interface with page navigation

---

## 📁 Project Structure
sales_forecasting_system/
│
├── data/
│ └── sales_data.csv
│
├── models/
│ └── (saved models: .pkl, .keras)
│
├── utils/
│ ├── data_loader.py
│ ├── recommendation_engine.py
│ ├── eda_tools.py
│ ├── metrics.py
│ └── swot_pipeline.py
│
├── streamlit_app/
│ ├── 1_home.py
│ └── pages/
│ ├── 2_Detailed_Dashboard.py
│ ├── 3_SWOT_Analysis.py
│ ├── 4_Recommendations.py
│ ├── 5_Forecast_ARIMA.py
│ ├── 6_Forecast_Prophet.py
│ ├── 7_Forecast_HoltWinters.py
│ ├── 8_Forecast_LSTM.py
│ ├── 9_Forecast_LightGBM.py
│ ├──10_Model_Comparison.py
│ ├──11_Train_Models.py
│ └──12_Model_Status
│
├── train_model.py
├── forecast_pipeline*.py (multiple models)
├── requirements.txt
└── README.md

---

## ▶️ Run the Project

```bash
# 1. Create environment (optional)
python -m venv venv
venv\Scripts\activate  # on Windows

# 2. Install requirements
pip install -r requirements.txt

# 3. Run Streamlit
streamlit run streamlit_app/1_Home.py


📊 Data Format
Ensure your CSV contains the following columns:

Date, Store ID, Product ID, Category, Region, Inventory Level,
Units Sold, Units Ordered, Demand Forecast, Price, Discount,
Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality

📩 Author
Developed by Yazan Noufal for a Master's Capstone Project – SVU 2025
🔥 Streamlit | AI Forecasting | Business Intelligenc

