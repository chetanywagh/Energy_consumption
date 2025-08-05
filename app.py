# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load XGBoost model
model = joblib.load("xgb_energy_forecast_model.joblib")

# Last known datetime in training dataset (change this if needed)
last_datetime = pd.to_datetime("2023-12-31 23:00:00")

# Streamlit UI
st.set_page_config(page_title="Energy Forecast", layout="wide")
st.title(" PJM Energy Forecast using XGBoost")
st.markdown("Predict future hourly energy consumption")

# Input days to forecast
n_days = st.slider("Select days to predict", 1, 30, 7)

# Generate future datetime
future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=n_days * 24, freq='H')
future_df = pd.DataFrame({'Datetime': future_dates})
future_df.set_index('Datetime', inplace=True)

# Feature engineering (match training phase)
future_df['hour'] = future_df.index.hour
future_df['dayofweek'] = future_df.index.dayofweek
future_df['month'] = future_df.index.month
future_df['day'] = future_df.index.day

# Select features
features = future_df[['hour', 'dayofweek', 'month', 'day']]

# Make predictions
future_df['Forecast_MW'] = model.predict(features)

# Plot predictions
st.subheader(" Forecasted Energy Usage")
fig, ax = plt.subplots(figsize=(12, 5))
future_df['Forecast_MW'].plot(ax=ax)
ax.set_title(f"Forecast for Next {n_days} Days")
ax.set_ylabel("MW")
st.pyplot(fig)

# Show data table
st.subheader(" Forecast Table (first 2 days)")
st.dataframe(future_df[['Forecast_MW']].head(48).style.format(precision=2))
