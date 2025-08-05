import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

# Title
st.title("🔌 PJM Hourly Energy Forecast")
st.markdown("Predict future energy consumption based on historical data.")

# Load model
try:
    with open("xgb_energy_forecast_model.joblib", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load past data
try:
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'], index_col='Datetime')
except Exception as e:
    st.error(f"❌ Error loading past data: {e}")
    st.stop()

# User input: number of days
days = st.slider("📅 Select number of future days to forecast:", 1, 30, 7)

# Feature Engineering Function
def create_features(data):
    data['lag_1'] = data['PJMW_MW'].shift(1)
    data['lag_2'] = data['PJMW_MW'].shift(2)
    data['rolling_mean_3'] = data['PJMW_MW'].rolling(window=3).mean()
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    return data

# Generate future dates
last_datetime = df.index[-1]
future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=days*24, freq='H')

# Combine with original for lag features
full_df = df.copy()

for future_date in future_dates:
    new_row = pd.DataFrame(index=[future_date])
    full_df = pd.concat([full_df, new_row])

    full_df = create_features(full_df)
    
    latest_row = full_df.loc[[future_date]].drop(columns=['PJMW_MW'], errors='ignore')
    latest_row = latest_row[['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']]

    if latest_row.isnull().any().any():
        full_df.loc[future_date, 'PJMW_MW'] = np.nan
        continue

    forecast_value = model.predict(latest_row)[0]
    full_df.loc[future_date, 'PJMW_MW'] = forecast_value

# Extract forecast part
forecast_df = full_df.loc[future_dates]
forecast_df = forecast_df[['PJMW_MW']].rename(columns={'PJMW_MW': 'Forecast_MW'})

# Plot
st.subheader(f"📊 Forecast for next {days} day(s)")
st.line_chart(forecast_df)

# Show data table
with st.expander("📄 See Forecasted Values"):
    st.dataframe(forecast_df.reset_index().rename(columns={'index': 'Datetime'}))
