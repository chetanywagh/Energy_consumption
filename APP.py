import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, time
import os
import warnings
import base64

warnings.filterwarnings('ignore')

st.set_page_config(page_title="ENERGY CONSUMPTION FORECAST", layout="centered")

# -------------------------
# Background Image
# -------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

img_base64 = get_base64_image("new image.jpeg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
                    url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: black;
    }}
    section[data-testid="stSidebar"] {{
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }}
    h1 {{ color: black !important; font-size: 25px !important; font-weight: 700 !important; }}
    h2, h3 {{ font-size: 20px !important; font-weight: 600 !important; }}
    p, label, div, span {{ font-size: 15px !important; color: black !important; }}
    section[data-testid="stSidebar"] * {{ font-size: 13px !important; }}
    .stDownloadButton button {{
        background-color: #81D4FA; color: white; border-radius: 8px; padding: 0.5rem 1rem;
    }}
    .stDownloadButton button:hover {{ background-color: #4FC3F7; }}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Energy Consumption Forecast")
st.markdown("""
This web app forecasts **daily energy consumption** (in MW) for the PJM region using a trained **XGBoost** model.

- Forecast start date: **2018-01-02**  
- Data is resampled from hourly to daily  
""")

# -------------------------
# Load Model & Data
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_energy_forecast_model.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
    df.set_index("Datetime", inplace=True)
    return df.resample("D").mean()

model = load_model()
data = load_data()

# -------------------------
# Feature Creation
# -------------------------
def create_features(df):
    df['lag_1'] = df['PJMW_MW'].shift(1)
    df['lag_2'] = df['PJMW_MW'].shift(2)
    df['rolling_mean_3'] = df['PJMW_MW'].rolling(window=3).mean().shift(1)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

# Match model's expected features
expected_features = model.get_booster().feature_names

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Forecast Settings")
start_date = datetime(2018, 1, 2).date()
st.sidebar.markdown(f"**Forecast Start Date:** `{start_date}`")

hourly_times = [time(h, 0) for h in range(24)]
start_time = st.sidebar.selectbox("Select Time (hourly):", hourly_times, index=0)
future_days = st.sidebar.slider("Days to Forecast:", 1, 50, 7)

# -------------------------
# Forecast Loop
# -------------------------
df = create_features(data.copy())
df.dropna(inplace=True)
predictions = []
last_known = df.copy()

with st.spinner(" Generating Forecast..."):
    for _ in range(future_days):
        next_date = last_known.index[-1] + timedelta(days=1)
        if next_date < pd.to_datetime(start_date):
            continue

        # Prepare next row
        next_row = pd.DataFrame(index=[next_date])
        next_row['lag_1'] = last_known['PJMW_MW'].iloc[-1]
        next_row['lag_2'] = last_known['PJMW_MW'].iloc[-2]
        next_row['rolling_mean_3'] = last_known['PJMW_MW'].iloc[-3:].mean()
        next_row['dayofweek'] = next_date.dayofweek
        next_row['month'] = next_date.month

        # Ensure order & all expected features
        for col in expected_features:
            if col not in next_row.columns:
                next_row[col] = 0  # fill missing with 0 or appropriate default
        X_pred = next_row[expected_features]

        pred = model.predict(X_pred)[0]
        next_row['PJMW_MW'] = pred

        last_known = pd.concat([last_known, next_row])
        predictions.append((datetime.combine(next_date.date(), start_time), pred))

# -------------------------
# Results
# -------------------------
forecast_df = pd.DataFrame(predictions, columns=["Datetime", "Forecast_MW"]).set_index("Datetime")
recent_actual = df[["PJMW_MW"]].rename(columns={"PJMW_MW": "Actual_MW"}).tail(30)
plot_df = pd.concat([recent_actual, forecast_df], axis=0)

# Plot
st.subheader("Energy Forecast Plot")
fig, ax = plt.subplots(figsize=(12, 5))
plot_df.plot(ax=ax, linewidth=2, marker='o', grid=True)
ax.set_xlabel("Date")
ax.set_ylabel("MW")
ax.set_title("Daily Energy Consumption Forecast")
fig.autofmt_xdate()
st.pyplot(fig)

# Summary
latest = forecast_df.Forecast_MW.values
st.markdown("### Forecast Summary")
col1, col2, col3 = st.columns(3)
col1.metric("🔺 Max Forecast", f"{np.max(latest):.2f} MW")
col2.metric("🔻 Min Forecast", f"{np.min(latest):.2f} MW")
col3.metric("📈 Avg Forecast", f"{np.mean(latest):.2f} MW")

# Table
st.subheader(f"Forecast Table - {future_days} Day(s)")
st.dataframe(forecast_df.reset_index().head(future_days))

# Download
st.download_button(
    label="Download Forecast CSV",
    data=forecast_df.reset_index().to_csv(index=False),
    file_name="daily_energy_forecast.csv",
    mime="text/csv"
)
