import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="PJM Energy Forecast Dashboard", layout="centered")
st.title("🔌 PJM Daily Energy Forecast Dashboard")
st.markdown("""
Welcome to the **PJM Energy Forecasting Dashboard**.

This professional application uses a trained **XGBoost** model to forecast daily energy consumption for the **PJM Interconnection**.

**Steps:**
- Select a forecast **start date**.
- Choose the **number of future days** you’d like to forecast (1–30).
- Visualize recent actuals and future forecasts.
- Download forecast data for reporting.
""")

# ----------------------
# Load Trained Model
# ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_energy_forecast_model.joblib")
    except Exception as e:
        st.error(f"\u274C Error loading model: {e}")
        st.stop()

model = load_model()

# ----------------------
# Load and Resample Data
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
        df.set_index("Datetime", inplace=True)
        return df.resample("D").mean()  # Daily average
    except Exception as e:
        st.error(f"\u274C Error loading dataset: {e}")
        st.stop()

data = load_data()

# ----------------------
# Feature Engineering
# ----------------------
def create_features(df):
    df['lag_1'] = df['PJMW_MW'].shift(1)
    df['lag_2'] = df['PJMW_MW'].shift(2)
    df['rolling_mean_3'] = df['PJMW_MW'].rolling(window=3).mean().shift(1)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

# ----------------------
# Sidebar Controls
# ----------------------
st.sidebar.header("🔧 Configuration")

future_days = st.sidebar.slider("Forecast horizon (days):", 1, 30, 7)

# Auto-detect last available date
last_available_date = data.index[-1].date()

forecast_start = st.sidebar.date_input(
    "Start forecast from:",
    value=last_available_date + timedelta(days=1),
    min_value=last_available_date + timedelta(days=1),
    max_value=last_available_date + timedelta(days=30)
)

# ----------------------
# Forecasting Logic
# ----------------------
df = data.copy()
df = create_features(df)
df.dropna(inplace=True)

# Get base data to forecast from selected date
base_df = df[df.index <= pd.to_datetime(forecast_start - timedelta(days=1))].copy()

if base_df.empty:
    st.warning("Not enough historical data before selected date to start forecasting.")
    st.stop()

predictions = []
last_known = base_df.copy()

for _ in range(future_days):
    next_date = last_known.index[-1] + timedelta(days=1)

    next_row = pd.DataFrame(index=[next_date])
    next_row['lag_1'] = last_known['PJMW_MW'].iloc[-1]
    next_row['lag_2'] = last_known['PJMW_MW'].iloc[-2]
    next_row['rolling_mean_3'] = last_known['PJMW_MW'].iloc[-3:].mean()
    next_row['dayofweek'] = next_date.dayofweek
    next_row['month'] = next_date.month

    X_pred = next_row[['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']]
    pred = model.predict(X_pred)[0]

    next_row['PJMW_MW'] = pred
    last_known = pd.concat([last_known, next_row])
    predictions.append((next_date, pred))

# ----------------------
# Results Preparation
# ----------------------
forecast_df = pd.DataFrame(predictions, columns=["Datetime", "Forecast_MW"]).set_index("Datetime")
recent_actual = df[["PJMW_MW"]].rename(columns={"PJMW_MW": "Actual_MW"}).tail(30)
plot_df = pd.concat([recent_actual, forecast_df], axis=0)


# ----------------------
# Visualization
# ----------------------
st.subheader(f"📊 Forecast from {forecast_start.strftime('%Y-%m-%d')} for {future_days} Days")
fig, ax = plt.subplots(figsize=(12, 5))
plot_df.plot(ax=ax, linewidth=2, marker='o')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("MW Consumption")
plt.title("PJM Daily Forecast vs Actual Energy Consumption")
plt.tight_layout()
plt.grid(True)
st.pyplot(fig)

# ----------------------
# 📄 Show Forecast Table
# ----------------------
st.markdown("### 📋 Forecasted Energy Consumption (MW)")
st.dataframe(
    forecast_df.reset_index().rename(columns={
        "Datetime": "Date",
        "Forecast_MW": "Forecasted Consumption (MW)"
    }),
    use_container_width=True
)

# ----------------------
# Download Forecast
# ----------------------
st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast_df.reset_index().to_csv(index=False),
    file_name="pjm_energy_forecast.csv",
    mime="text/csv"
)

# ----------------------
# Download Forecast
# ----------------------
st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast_df.reset_index().to_csv(index=False),
    file_name="pjm_energy_forecast.csv",
    mime="text/csv"
)
