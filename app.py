import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.title("📡 Unified Network Management System")
st.subheader("AI/ML-Based Predictive Outage Detection")

# -----------------------------
# Step 1: Dummy Data Generation
# -----------------------------
@st.cache_resource
def train_models():
    data = pd.DataFrame({
        'latency': np.random.randint(10, 300, 500),
        'packet_loss': np.random.randint(0, 50, 500),
        'bandwidth': np.random.randint(50, 1000, 500),
        'failure': np.random.randint(0, 2, 500)
    })

    X = data[['latency', 'packet_loss', 'bandwidth']]
    y = data['failure']

    # Classification model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Anomaly detection
    anomaly_model = IsolationForest(contamination=0.05)
    anomaly_model.fit(X)

    return model, anomaly_model

model, anomaly_model = train_models()

# -----------------------------
# Step 2: User Input
# -----------------------------
st.sidebar.header("Enter Network Parameters")

latency = st.sidebar.slider("Latency (ms)", 10, 500, 50)
packet_loss = st.sidebar.slider("Packet Loss (%)", 0, 100, 5)
bandwidth = st.sidebar.slider("Bandwidth (Mbps)", 10, 1000, 100)

input_data = np.array([[latency, packet_loss, bandwidth]])

# -----------------------------
# Step 3: Prediction
# -----------------------------
if st.button("🔍 Predict Network Status"):

    prediction = model.predict(input_data)[0]
    anomaly = anomaly_model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Network Failure")
    else:
        st.success("✅ Network is Stable")

    if anomaly == -1:
        st.warning("🚨 Anomaly Detected (Unusual Pattern Found)")
    else:
        st.info("✔️ No anomaly detected")

# -----------------------------
# Info Section
# -----------------------------
st.markdown("---")
st.markdown("### ℹ️ About Project")
st.write("""
This system predicts network failures using:
- Random Forest (classification)
- Isolation Forest (anomaly detection)

It helps in proactive outage detection instead of reactive troubleshooting.
""")