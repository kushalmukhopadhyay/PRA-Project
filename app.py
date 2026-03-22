# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import matplotlib.pyplot as plt

st.title("🧠 Smart Health Monitoring System")

# ---- Simulate Data ----
n = 500
data = pd.DataFrame({
    "heart_rate": np.random.normal(75, 10, n),
    "steps": np.random.normal(8000, 2000, n),
    "sleep_hours": np.random.normal(7, 1.5, n),
})
data["risk"] = ((data["heart_rate"] > 100) | 
                (data["sleep_hours"] < 5) |
                (data["steps"] < 3000)).astype(int)

iso = IsolationForest(contamination=0.05)
data["anomaly"] = iso.fit_predict(data[["heart_rate", "steps", "sleep_hours"]])
data["anomaly"] = data["anomaly"].map({1:0, -1:1})

# ---- Train Model ----
X = data[["heart_rate","steps","sleep_hours"]]
y = data["risk"]
model = RandomForestClassifier()
model.fit(X, y)

# ---- User Input ----
st.sidebar.header("Input Your Health Data")
heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 80)
steps = st.sidebar.slider("Steps", 0, 20000, 8000)
sleep = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)

input_data = np.array([[heart_rate, steps, sleep]])
prediction = model.predict(input_data)

# ---- Display Result ----
st.header("🩺 Health Prediction")
if prediction[0] == 1:
    st.error("⚠️ High Health Risk Detected!")
else:
    st.success("✅ Normal Health Condition")

# ---- Visualization ----
st.header("📊 Health Patterns")
fig, ax = plt.subplots()
ax.plot(data["heart_rate"], label="Heart Rate")
ax.plot(data["sleep_hours"]*10, label="Sleep (scaled)")
ax.legend()
st.pyplot(fig)
