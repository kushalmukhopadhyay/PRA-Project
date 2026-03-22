# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Health Monitoring", layout="wide")
st.title("🧠 Smart Health Monitoring System")

# ---------------------------
# 1️⃣ Modular Inputs
# ---------------------------
st.sidebar.header("Enter Your Health Data")
metrics = {
    "Heart Rate (BPM)": st.sidebar.slider("Heart Rate", 50, 150, 80),
    "Steps": st.sidebar.slider("Steps", 0, 20000, 8000),
    "Sleep Hours": st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0),
    "SpO2 (%)": st.sidebar.slider("Blood Oxygen", 80, 100, 98),
    "Body Temp (°C)": st.sidebar.slider("Body Temperature", 35.0, 40.0, 36.6),
    "Stress Level (0-10)": st.sidebar.slider("Stress Level", 0, 10, 3)
}

# Convert to DataFrame for analysis
user_df = pd.DataFrame([metrics])

# ---------------------------
# 2️⃣ Statistical Summary
# ---------------------------
st.subheader("📊 Statistical Summary")
st.write(user_df.describe())

# ---------------------------
# 3️⃣ Anomaly Detection
# ---------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
# Fit on user data for demo; ideally fit on historical dataset
iso.fit(user_df)
anomaly = iso.predict(user_df)
if anomaly[0] == -1:
    st.warning("⚠️ One or more metrics are abnormal!")
else:
    st.success("✅ All metrics are within normal range")

# ---------------------------
# 4️⃣ Risk Scoring
# ---------------------------
weights = {
    "Heart Rate (BPM)": 0.2,
    "Steps": 0.15,
    "Sleep Hours": 0.2,
    "SpO2 (%)": 0.2,
    "Body Temp (°C)": 0.15,
    "Stress Level (0-10)": 0.1
}

# Define simple thresholds
risk_flags = {
    "Heart Rate (BPM)": lambda x: x>100 or x<50,
    "Steps": lambda x: x<3000,
    "Sleep Hours": lambda x: x<5,
    "SpO2 (%)": lambda x: x<94,
    "Body Temp (°C)": lambda x: x>37.5,
    "Stress Level (0-10)": lambda x: x>7
}

risk_score = sum(weights[m] for m,v in metrics.items() if risk_flags[m](v))
st.subheader("🩺 Health Risk Score")
st.write(f"{risk_score*100:.1f}%")
if risk_score > 0.3:
    st.error("⚠️ High Risk Detected!")
else:
    st.success("✅ Low Risk")

# ---------------------------
# 5️⃣ Trend Visualization (Simulated)
# ---------------------------
st.subheader("📈 Trend Analysis")
# Simulate historical data for demo
history = pd.DataFrame({
    "Heart Rate": np.random.normal(75, 10, 7),
    "Steps": np.random.normal(8000, 2000, 7),
    "Sleep Hours": np.random.normal(7, 1.5, 7),
    "SpO2 (%)": np.random.normal(98, 1, 7),
    "Body Temp (°C)": np.random.normal(36.6, 0.5, 7),
    "Stress Level (0-10)": np.random.randint(1,5,7)
})

fig, ax = plt.subplots(figsize=(10,5))
for col in history.columns:
    ax.plot(history[col], marker='o', label=col)
ax.set_xlabel("Days")
ax.set_ylabel("Values")
ax.set_title("Health Metrics Trend Over 7 Days")
ax.legend()
st.pyplot(fig)

# ---------------------------
# 6️⃣ Optional: Show Raw Data
# ---------------------------
st.subheader("🗂 Raw Data (Current + Simulated History)")
st.write(pd.concat([user_df, history], ignore_index=True))
