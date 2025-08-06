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

# Copy original dataframe
full_df = df.copy()

# Generate future dates
last_datetime = full_df.index[-1]
future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=days * 24, freq='H')

# Forecast list to store results
forecast_values = []

for future_date in future_dates:
    # Create empty row
    new_row = pd.DataFrame(index=[future_date])
    full_df = pd.concat([full_df, new_row])

    # Apply feature engineering on updated dataframe
    full_df = create_features(full_df)

    # Get latest row for prediction
    latest_row = full_df.loc[[future_date]]
    
    # Select only required features
    try:
        features = latest_row[['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']]
    except KeyError:
        forecast_values.append(np.nan)
        continue

    # Handle missing values
    if features.isnull().values.any():
        forecast_values.append(np.nan)
        continue

    # Predict using model
    prediction = model.predict(features)[0]
    
    # Store prediction
    full_df.at[future_date, 'PJMW_MW'] = prediction
    forecast_values.append(prediction)

# Prepare forecast dataframe
forecast_df = pd.DataFrame({
    'Datetime': future_dates,
    'Forecast_MW': forecast_values
}).set_index('Datetime')

# Plot
st.subheader(f"📊 Forecast for next {days} day(s)")
st.line_chart(forecast_df['Forecast_MW'])

# Show forecast table
with st.expander("📄 See Forecasted Values"):
    st.dataframe(forecast_df.reset_index())
