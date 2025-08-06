import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, time

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="PJM Daily Energy Forecast", layout="centered")
st.title("PJM Daily Energy Forecast")
st.markdown("""
This professional web app forecasts PJM **daily** energy consumption using XGBoost.
You can select the forecast start date, duration, and view automatic insights below the graph.
""")

# ----------------------
# Load Trained Model
# ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_energy_forecast_model.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# ----------------------
# Load Past Data
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
        df.set_index("Datetime", inplace=True)
        daily_df = df.resample("D").mean()
        return daily_df
    except Exception as e:
        st.error(f"Error loading past data: {e}")
        st.stop()

data = load_data()

# ----------------------
# Feature Engineering for Daily Forecasting
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
st.sidebar.header("Forecast Settings")
default_start_date = data.index[-1] + timedelta(days=1)
start_date = st.sidebar.date_input("Forecast Start Date:", default_start_date)
start_time = st.sidebar.time_input("Start Time (for record):", value=time(0, 0))
future_days = st.sidebar.slider("Forecast days:", min_value=1, max_value=30, value=7)

# ----------------------
# Forecasting Logic
# ----------------------
df = data.copy()
df = create_features(df)
df.dropna(inplace=True)

predictions = []
last_known = df.copy()

for i in range(future_days):
    next_date = last_known.index[-1] + timedelta(days=1)
    if next_date < pd.to_datetime(start_date):
        next_row = pd.DataFrame(index=[next_date])
        next_row['PJMW_MW'] = np.nan
        last_known = pd.concat([last_known, next_row])
        continue

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
    predictions.append((datetime.combine(next_date.date(), start_time), pred))

# ----------------------
# Prepare Output
# ----------------------
forecast_df = pd.DataFrame(predictions, columns=["Datetime", "Forecast_MW"]).set_index("Datetime")
recent_actual = df[["PJMW_MW"]].rename(columns={"PJMW_MW": "Actual_MW"}).tail(30)
plot_df = pd.concat([recent_actual, forecast_df], axis=0)

# ----------------------
# Plot
# ----------------------
st.subheader("Forecasted Energy Consumption")
fig, ax = plt.subplots(figsize=(12, 5))
plot_df.plot(ax=ax, linewidth=2, marker='o')
ax.set_xlabel("Datetime")
ax.set_ylabel("MW Consumption")
ax.set_title("Energy Consumption Forecast")
ax.grid(True)
fig.autofmt_xdate()

for date in forecast_df.index:
    ax.axvline(x=date, color='gray', linestyle='--', linewidth=0.4)

st.pyplot(fig)

# ----------------------
# Auto Graph Summary Below Plot
# ----------------------
latest = forecast_df.Forecast_MW.values
max_val = np.max(latest)
min_val = np.min(latest)
avg_val = np.mean(latest)

st.markdown("""
### 📊 Automatic Graph Summary
- **Maximum Forecasted Consumption**: {:.2f} MW  
- **Minimum Forecasted Consumption**: {:.2f} MW  
- **Average Forecasted Consumption**: {:.2f} MW
""".format(max_val, min_val, avg_val))

# ----------------------
# Download Option
# ----------------------
st.download_button("📥 Download Forecast Data as CSV",
                   data=forecast_df.reset_index().to_csv(index=False),
                   file_name="daily_forecast.csv",
                   mime="text/csv")


