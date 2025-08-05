# energy_forecast_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# --------------------- Page Setup ---------------------
st.set_page_config(page_title="🔌 PJM Energy Forecast App", layout="wide")
st.title("🔋 PJM Hourly Energy Consumption Forecast")
st.markdown("Predict hourly energy usage for the next 7 days using a machine learning model.")

# --------------------- Load Model ---------------------
try:
    model = joblib.load("final_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --------------------- Load Historical Data ---------------------
try:
    data = pd.read_csv("past_data.csv", parse_dates=['Datetime'], index_col='Datetime')
    st.success("✅ Historical data loaded!")
except Exception as e:
    st.error(f"❌ Error loading past data: {e}")
    st.stop()

# --------------------- Generate Future Dates ---------------------
last_datetime = data.index[-1]
future_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=24 * 7, freq='H')
future_df = pd.DataFrame(index=future_dates)

# --------------------- Feature Engineering ---------------------
try:
    # Use latest values for lag features
    lag_1_value = data['PJMW_MW'].iloc[-1]
    lag_2_value = data['PJMW_MW'].iloc[-2]
    rolling_3_value = data['PJMW_MW'].iloc[-3:].mean()

    # Assign these values to all future rows
    future_df['lag_1'] = lag_1_value
    future_df['lag_2'] = lag_2_value
    future_df['rolling_mean_3'] = rolling_3_value
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month

    # Prepare features in correct order
    feature_columns = ['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']
    future_features = future_df[feature_columns]

    # Predict
    future_df['Forecast_MW'] = model.predict(future_features)

except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

# --------------------- Display Forecast ---------------------
st.subheader("📊 7-Day Hourly Forecast")
st.line_chart(future_df['Forecast_MW'])

# --------------------- Download Option ---------------------
csv_data = future_df.reset_index()[['Datetime', 'Forecast_MW']].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Forecast Data (CSV)", data=csv_data, file_name='forecast.csv', mime='text/csv')
