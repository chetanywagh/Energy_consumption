import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="PJM Energy Forecast", layout="centered")
st.title("🔌 PJM Hourly Energy Forecast")
st.markdown("""
This web app allows you to forecast PJM hourly energy consumption.
Select the number of future days you'd like to forecast, and see the prediction plotted with recent data.
""")

# ----------------------
# Load Trained Model
# ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_energy_forecast_model.joblib")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ----------------------
# Load Past Data
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'])
        df.set_index('Datetime', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error loading past data: {e}")
        st.stop()

data = load_data()

# ----------------------
# Feature Engineering Function
# ----------------------
def create_features(df):
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df[['hour', 'day', 'dayofweek', 'month']]

# ----------------------
# User Input for Forecast
# ----------------------
future_days = st.slider("Select how many future days to forecast:", min_value=1, max_value=30, value=7)

# ----------------------
# Prepare Future Dates
# ----------------------
last_datetime = data.index[-1]
future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=future_days*24, freq='H')
future_df = pd.DataFrame(index=future_dates)

# ----------------------
# Feature Engineering for Future Data
# ----------------------
future_features = create_features(future_df.copy())

# ----------------------
# Forecasting
# ----------------------
try:
    forecast = model.predict(future_features)
    future_df['Forecast_MW'] = forecast
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

# ----------------------
# Combine with Past Data
# ----------------------
combined_df = pd.concat([
    data[['PJMW_MW']].rename(columns={'PJMW_MW': 'Actual_MW'}).tail(24*7),
    future_df[['Forecast_MW']]
])

# ----------------------
# Plot
# ----------------------
st.subheader(f"📈 Forecast for Next {future_days} Days")
fig, ax = plt.subplots(figsize=(10, 4))
combined_df.plot(ax=ax, linewidth=2)
plt.xlabel("Datetime")
plt.ylabel("MW Consumption")
plt.title("PJM Energy Forecast")
plt.grid(True)
st.pyplot(fig)

# ----------------------
# Download Option
# ----------------------
st.download_button("📥 Download Forecast Data as CSV", data=future_df.to_csv(), file_name="forecast.csv")
