import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="⚡ PJM Hourly Energy Forecast", layout="centered")

st.title("⚡ PJM Hourly Energy Forecast App")
st.markdown("Forecast hourly energy consumption for the PJMW region.")

# ==================== Load Model ====================
try:
    model = joblib.load("final_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ==================== Load Historical Data ====================
try:
    data = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'])
    data.set_index('Datetime', inplace=True)
except Exception as e:
    st.error(f"❌ Error loading past data: {e}")
    st.stop()

# ==================== Show Last 7 Days of Data ====================
st.subheader("🔍 Last 7 Days Energy Usage")
st.line_chart(data.tail(7 * 24)['PJMW_MW'])

# ==================== User Input for Forecast ====================
forecast_days = st.slider("📅 Select number of future days to forecast:", min_value=1, max_value=30, value=7)

# ==================== Feature Engineering ====================
last_datetime = data.index[-1]
future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=forecast_days * 24, freq='H')
future_df = pd.DataFrame(index=future_dates)
future_df.index.name = 'Datetime'

# Generate lag & rolling features based on last known data
merged_df = pd.concat([data, future_df])
merged_df['lag_1'] = merged_df['PJMW_MW'].shift(1)
merged_df['lag_2'] = merged_df['PJMW_MW'].shift(2)
merged_df['rolling_mean_3'] = merged_df['PJMW_MW'].shift(1).rolling(window=3).mean()

# Time-based features
merged_df['hour'] = merged_df.index.hour
merged_df['day'] = merged_df.index.day
merged_df['dayofweek'] = merged_df.index.dayofweek
merged_df['month'] = merged_df.index.month

# Extract only future rows
future_features = merged_df.loc[future_df.index][['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']]

# Drop rows with NaN (first few hours will have)
future_features.dropna(inplace=True)

# ==================== Make Forecast ====================
try:
    forecast = model.predict(future_features)
    future_df = future_df.loc[future_features.index]  # align index
    future_df['Forecast_MW'] = forecast
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

# ==================== Show Forecast ====================
st.subheader(f"📈 Forecast for Next {forecast_days} Days")
st.line_chart(future_df['Forecast_MW'])

# ==================== Plot Combined Graph ====================
st.subheader("📊 Past vs Forecast Graph")
combined = pd.concat([data[['PJMW_MW']], future_df[['Forecast_MW']]], axis=0)
st.line_chart(combined.rename(columns={'PJMW_MW': 'Actual', 'Forecast_MW': 'Forecast'}))
