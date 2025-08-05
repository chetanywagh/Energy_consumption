# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ---------------------------- CONFIGURATION ---------------------------- #
st.set_page_config(page_title="PJM Energy Forecast", layout="wide")
st.title("🔌 PJM Hourly Energy Consumption Forecast")

# ---------------------------- MODEL LOADING ---------------------------- #
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgb_energy_forecast_model.joblib")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()
if not model:
    st.stop()

# ---------------------------- DATA LOADING ---------------------------- #
@st.cache_data
def load_past_data():
    try:
        df = pd.read_csv("C:\Users\Lenovo\OneDrive\Desktop\project Deployment Excelr\new project 2 ExcelR\PJMW_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
        return df
    except Exception as e:
        st.error(f"❌ Error loading past data: {e}")
        return None

data = load_past_data()
if data is None:
    st.stop()

# ---------------------------- USER INPUT ---------------------------- #
st.sidebar.header("🔧 Forecast Settings")
forecast_days = st.sidebar.slider("Select forecast duration (days)", 1, 7, 3)

# ---------------------------- FEATURE ENGINEERING ---------------------------- #
def generate_features(data, forecast_hours):
    last_datetime = data.index[-1]
    future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=forecast_hours, freq='H')
    future_df = pd.DataFrame(index=future_dates)

    # Lag and rolling from past data
    future_df['lag_1'] = data['PJMW_MW'].iloc[-1]
    future_df['lag_2'] = data['PJMW_MW'].iloc[-2]
    future_df['rolling_mean_3'] = data['PJMW_MW'].rolling(window=3).mean().iloc[-1]

    # Time-based features
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month

    # Fill missing values forward
    for col in ['lag_1', 'lag_2', 'rolling_mean_3']:
        future_df[col] = future_df[col].fillna(method='ffill')

    return future_df

forecast_hours = forecast_days * 24
future_df = generate_features(data, forecast_hours)

# ---------------------------- PREDICTION ---------------------------- #
try:
    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']
    X_future = future_df[feature_cols]
    future_df['Forecast_MW'] = model.predict(X_future)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

# ---------------------------- OUTPUT DISPLAY ---------------------------- #
st.subheader(f"📊 Forecast for Next {forecast_days} Days (Hourly)")
st.line_chart(future_df['Forecast_MW'])

# ---------------------------- DOWNLOAD OPTION ---------------------------- #
download_df = future_df.reset_index().rename(columns={"index": "Datetime"})
csv_data = download_df[['Datetime', 'Forecast_MW']].to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Forecast as CSV",
    data=csv_data,
    file_name="pjm_forecast.csv",
    mime="text/csv"
)

