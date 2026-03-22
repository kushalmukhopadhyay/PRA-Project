import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Smart Health Monitoring System", layout="wide")
st.title("🧠 Smart Health Monitoring System (ML Integrated)")

# -----------------------------
# Load real wearable dataset
# -----------------------------
historical_df = pd.read_csv("wearable_data.csv")  # Your real dataset

# -----------------------------
# Train IsolationForest
# -----------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(historical_df)

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Enter Your Health Data")
heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 80)
steps = st.sidebar.slider("Steps", 0, 20000, 8000)
sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
spo2 = st.sidebar.slider("SpO2 (%)", 80, 100, 98)
body_temp = st.sidebar.slider("Body Temperature", 35.0, 40.0, 36.6)
stress_level = st.sidebar.slider("Stress Level (0-10)", 0, 10, 3)

user_df = pd.DataFrame([[heart_rate, steps, sleep_hours, spo2, body_temp, stress_level]],
                       columns=["heart_rate","steps","sleep_hours","spo2","body_temp","stress_level"])

# -----------------------------
# Predict anomaly
# -----------------------------
anomaly = iso.predict(user_df)[0]

# -----------------------------
# Risk scoring
# -----------------------------
weights = {
    "heart_rate": 0.2,
    "steps": 0.15,
    "sleep_hours": 0.2,
    "spo2": 0.2,
    "body_temp": 0.15,
    "stress_level": 0.1
}

risk_flags = {
    "heart_rate": lambda x: x>100 or x<50,
    "steps": lambda x: x<3000,
    "sleep_hours": lambda x: x<5,
    "spo2": lambda x: x<94,
    "body_temp": lambda x: x>37.5,
    "stress_level": lambda x: x>7
}

metrics = user_df.iloc[0].to_dict()
risk_score = sum(weights[m] for m,v in metrics.items() if risk_flags[m](v))

# -----------------------------
# Display results
# -----------------------------
if anomaly == -1:
    st.warning("⚠️ One or more metrics are abnormal!")
else:
    st.success("✅ All metrics are normal")

st.subheader("🩺 Health Risk Score")
st.write(f"{risk_score*100:.2f}%")

# -----------------------------
# Simulated 7-day trends
# -----------------------------
trend = pd.DataFrame({
    "heart_rate": np.random.normal(heart_rate, 5, 7),
    "steps": np.random.normal(steps, 1000, 7),
    "sleep_hours": np.random.normal(sleep_hours, 1, 7),
    "spo2": np.random.normal(spo2, 1, 7),
    "body_temp": np.random.normal(body_temp, 0.3, 7),
    "stress_level": np.random.normal(stress_level, 1, 7)
})

st.subheader("📈 Simulated 7-Day Trend")
fig, ax = plt.subplots(figsize=(10,5))
for col in trend.columns:
    ax.plot(trend[col], marker='o', label=col)
ax.legend()
st.pyplot(fig)
