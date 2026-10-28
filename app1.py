import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Household Electric Power Predictor",
    page_icon="⚡",
    layout="wide"
)

# Load trained models, feature names, and metadata safely
@st.cache_resource
def load_artifacts():
    # Adjust paths if your app.py is inside a subdirectory relative to models/ and data/
    model_path = "models/models.pkl"
    comp_path = "models/model_comparison.csv"
    live_path = "data/processed/live_sample.csv"
    
    # Fallback paths check for relative directory nesting
    if not os.path.exists(model_path):
        model_path = "../models/models.pkl"
        comp_path = "../models/model_comparison.csv"
        live_path = "../data/processed/live_sample.csv"

    artifacts = joblib.load(model_path)
    comparison = pd.read_csv(comp_path, index_col=0)
    live_sample = pd.read_csv(live_path, parse_dates=True, index_col=0)
    return artifacts, comparison, live_sample

try:
    artifacts, comparison, live_sample = load_artifacts()
    fitted_models = artifacts["models"]
    features = artifacts["features"]
    best_name = artifacts["best"]
    meta = artifacts["meta"]
except Exception as e:
    st.error(f"⚠️ Could not load model artifacts. Ensure you have run the training notebook and saved files properly. Error: {e}")
    st.stop()

# App Header
st.title("⚡ Household Electric Power Consumption Dashboard")
st.markdown("Predict household energy usage in real time and analyze model performance benchmarks.")

# Sidebar Controls for Input Features
st.sidebar.header("🎛️ Interactive Controls")

model_choice = st.sidebar.selectbox(
    "Select Regression Model",
    options=list(fitted_models.keys()),
    index=list(fitted_models.keys()).index(best_name) if best_name in fitted_models else 0
)

st.sidebar.subheader("Appliance Sub-Metering")
sub1 = st.sidebar.slider("Sub Metering 1 (Kitchen Wh)", 0.0, meta["sub1_max"], 0.0, help="Dishwasher, oven, microwave")
sub2 = st.sidebar.slider("Sub Metering 2 (Laundry Wh)", 0.0, meta["sub2_max"], 0.0, help="Washing machine, tumble dryer, fridge")
sub3 = st.sidebar.slider("Sub Metering 3 (Water/AC Wh)", 0.0, meta["sub3_max"], 0.0, help="Electric water heater, air conditioning")

st.sidebar.subheader("Grid & Temporal Settings")
voltage = st.sidebar.slider("Voltage (V)", meta["volt_min"], meta["volt_max"], 240.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
)
is_weekend = 1 if day_of_week >= 5 else 0
month = st.sidebar.slider("Month", 1, 12, 6)

# Process features into exact model input format
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

input_dict = {
    "Sub_metering_1": sub1,
    "Sub_metering_2": sub2,
    "Sub_metering_3": sub3,
    "Voltage": voltage,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "month": month
}

# Construct DataFrame matching training feature columns order
input_df = pd.DataFrame([input_dict])[features]

# Make Prediction
selected_model = fitted_models[model_choice]
prediction = selected_model.predict(input_df)[0]

# Main Dashboard Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Real-Time Prediction")
    st.metric(
        label=f"Predicted Global Active Power ({model_choice})",
        value=f"{max(0.0, prediction):.3f} kW"
    )
    
    st.markdown("### 📊 Sub-Metering Proportions")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Estimating proportional weights for visualization
    sub_total_kw = (sub1 + sub2 + sub3) / 1000.0
    other_kw = max(0.1, prediction - sub_total_kw)
    sizes = [max(0.1, sub1), max(0.1, sub2), max(0.1, sub3), max(0.1, other_kw * 1000)]
    labels = ['Kitchen (Sub 1)', 'Laundry (Sub 2)', 'Water/AC (Sub 3)', 'Other / Unmetered']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

with col2:
    st.subheader("📈 Model Performance Benchmarks")
    st.dataframe(comparison.style.highlight_min(subset=["RMSE", "MAE"], color="lightgreen"))
    
    st.subheader("Live Test Dataset Preview")
    st.write(f"Loaded continuous minute timeline: **{len(live_sample):,} rows** available.")
    if st.checkbox("Preview raw live sample data frame"):
        st.dataframe(live_sample.head(10))