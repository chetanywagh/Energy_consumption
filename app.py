import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, time
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------
# Page Config + Custom Styling
# -------------------------------------
st.set_page_config(page_title="PJM Daily Energy Forecast", layout="centered")

st.markdown("""
    <style>
    /* Background Gradient */
    .reportview-container {
        background: linear-gradient(135deg, #e9f1f7 0%, #fefefe 100%);
    }

    /* White stylish sidebar */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }

    /* Title and sliders */
    h1 {
        color: #0A5275;
    }

    .stSlider > div[data-baseweb="slider"] > div {
        background-color: #0A5275;
    }

    /* Buttons */
    .stDownloadButton button {
        background-color: #0A5275;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stDownloadButton button:hover {
        background-color: #06394f;
        color: white;
    }

    /* Metric style */
    .element-container:has(div[data-testid="stMetric"]) p {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Logo (optional)
# -------------------------
st.image("cf1c74d9-df02-4984-afdf-f3c1cc2cc8af.png", width=100)

# -------------------------
# Title & Introduction
# -------------------------
st.title("🔌 PJM Daily Energy Forecast")

st.markdown("""
This professional web application forecasts **daily energy consumption** (in MW) for the PJM region using a trained **XGBoost** model.

- 📅 Forecast start date is fixed at **2018-01-02**
- 📊 Data is resampled from hourly to daily granularity
""")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_energy_forecast_model.joblib")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
        df.set_index("Datetime", inplace=True)
        daily_df = df.resample("D").mean()
        return daily_df
    except Exception as e:
        st.error(f"❌ Error loading past data: {e}")
        st.stop()

data = load_data()

# -------------------------
# Feature Engineering
# -------------------------
def create_features(df):
    df['lag_1'] = df['PJMW_MW'].shift(1)
    df['lag_2'] = df['PJMW_MW'].shift(2)
    df['rolling_mean_3'] = df['PJMW_MW'].rolling(window=3).mean().shift(1)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("🛠️ Forecast Settings")

start_date = datetime(2018, 1, 2).date()
st.sidebar.markdown("**Forecast Start Date:**")
st.sidebar.markdown(f"📅 `{start_date}` (fixed)")

hourly_times = [time(h, 0) for h in range(24)]
start_time = st.sidebar.selectbox("Select Time (hourly):", hourly_times, index=0)

future_days = st.sidebar.slider("Days to Forecast:", min_value=1, max_value=50, value=7)

# -------------------------
# Forecast Logic
# -------------------------
df = data.copy()
df = create_features(df)
df.dropna(inplace=True)

predictions = []
last_known = df.copy()

for i in range(future_days):
    next_date = last_known.index[-1] + timedelta(days=1)
    if next_date < pd.to_datetime(start_date):
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

# -------------------------
# Output Plot Data
# -------------------------
forecast_df = pd.DataFrame(predictions, columns=["Datetime", "Forecast_MW"]).set_index("Datetime")
recent_actual = df[["PJMW_MW"]].rename(columns={"PJMW_MW": "Actual_MW"}).tail(30)
plot_df = pd.concat([recent_actual, forecast_df], axis=0)

# -------------------------
# Plot
# -------------------------
st.subheader("📉 Energy Forecast Plot")

fig, ax = plt.subplots(figsize=(12, 5))
plot_df.plot(ax=ax, linewidth=2, marker='o', grid=True)
ax.set_xlabel("Date")
ax.set_ylabel("MW")
ax.set_title("Daily Energy Consumption Forecast")
fig.autofmt_xdate()

st.pyplot(fig)

# -------------------------
# Summary Section
# -------------------------
latest = forecast_df.Forecast_MW.values
max_val = np.max(latest)
min_val = np.min(latest)
avg_val = np.mean(latest)

st.markdown("### 📊 Forecast Summary")
col1, col2, col3 = st.columns(3)
col1.metric("🔺 Max Forecast", f"{max_val:.2f} MW")
col2.metric("🔻 Min Forecast", f"{min_val:.2f} MW")
col3.metric("📈 Avg Forecast", f"{avg_val:.2f} MW")

# -------------------------
# Download Button
# -------------------------
st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast_df.reset_index().to_csv(index=False),
    file_name="daily_energy_forecast.csv",
    mime="text/csv"
)
