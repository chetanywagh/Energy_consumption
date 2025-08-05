import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Energy Forecast", layout="wide")
st.title("⚡ PJM Energy Forecast using XGBoost")
st.markdown("Predict hourly energy usage for next 1–30 days")

try:
    model = joblib.load("xgb_energy_forecast_model.joblib")  # Must match training name
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


n_days = st.slider("Select number of days to predict", min_value=1, max_value=30, value=7)


last_datetime = pd.to_datetime("2023-12-31 23:00:00")


future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=n_days * 24, freq='H')
future_df = pd.DataFrame({'Datetime': future_dates})
future_df.set_index('Datetime', inplace=True)

future_df['hour'] = future_df.index.hour
future_df['dayofweek'] = future_df.index.dayofweek
future_df['month'] = future_df.index.month
future_df['day'] = future_df.index.day

# Adjust column names as per training data
features = future_df[['hour', 'dayofweek', 'month', 'day']]  # Must match order + names

try:
    future_df['Forecast_MW'] = model.predict(features)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

st.subheader(" Forecast Plot")
fig, ax = plt.subplots(figsize=(12, 5))
future_df['Forecast_MW'].plot(ax=ax)
ax.set_title(f"Forecast for Next {n_days} Days")
ax.set_ylabel("Energy Usage (MW)")
st.pyplot(fig)

st.subheader("🔢 Forecast Table (First 48 Hours)")
st.dataframe(future_df[['Forecast_MW']].head(48).style.format(precision=2))
