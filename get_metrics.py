# get_metrics.py
import pandas as pd

# Load the data from the CSV file
# تأكد من أن اسم الملف هو 'sales_data.csv'
try:
    df = pd.read_csv('data\sales_data.csv')
except FileNotFoundError:
    print("Error: The file 'sales_data.csv' was not found.")
    exit()

# Ensure 'Date' column is a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# --- Calculate the metrics ---

# Daily Demand
daily_demand = df.groupby('Date')['Units Ordered'].sum()
avg_daily_demand = daily_demand.mean()
median_daily_demand = daily_demand.median()

# Product Sales
product_sales = df.groupby('Product ID')['Units Sold'].sum()
avg_product_sales = product_sales.mean()
median_product_sales = product_sales.median()

# Print the results
print("--- Sales & Demand Metrics ---")
print(f"Average Daily Demand: {avg_daily_demand:,.2f}")
print(f"Median Daily Demand: {median_daily_demand:,.2f}")
print("-" * 20)
print(f"Average Product Sales: {avg_product_sales:,.2f}")
print(f"Median Product Sales: {median_product_sales:,.2f}")
print("-" * 20)
print(f"Total Unique Products: {df['Product ID'].nunique()}")
print("-" * 20)