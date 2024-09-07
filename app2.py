import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("DWLR_Dataset_2023.csv")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Function to calculate Z-Score and detect anomalies
def calculate_z_scores(feature):
    data[f'{feature}_Z-Score'] = stats.zscore(data[feature])
    data[f'{feature}_Anomaly'] = np.where(np.abs(data[f'{feature}_Z-Score']) > 3, 1, 0)

# Features to analyze
features = ['Water_Level_m', 'Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']

# Apply Z-Score calculation and anomaly detection
for feature in features:
    calculate_z_scores(feature)

# Streamlit UI components
st.title("Dynamic Anomaly Detection App")
st.write("This app dynamically plots graphs for selected features and detects anomalies using Z-Score.")

# Sidebar for feature selection
selected_feature = st.sidebar.selectbox("Select Feature to Plot", features)

# Display selected feature's data
st.write(f"Displaying data for {selected_feature}")
st.write(data[['Date', selected_feature, f'{selected_feature}_Z-Score', f'{selected_feature}_Anomaly']])

# Plotting
st.subheader(f'{selected_feature} Over Time with Anomalies Highlighted')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data['Date'], data[selected_feature], label=selected_feature, color='blue')
ax.scatter(data[data[f'{selected_feature}_Anomaly'] == 1]['Date'], 
           data[data[f'{selected_feature}_Anomaly'] == 1][selected_feature],
           color='red', label='Anomalies', marker='o')
ax.set_title(f'{selected_feature} Over Time with Anomalies Highlighted')
ax.set_xlabel('Date')
ax.set_ylabel(selected_feature)
ax.legend()
ax.grid()
st.pyplot(fig)

# Alert system for anomalies
if data[f'{selected_feature}_Anomaly'].sum() > 0:
    st.error(f"Anomalies detected in {selected_feature}!")

st.sidebar.write("Use this sidebar to select different features and view the corresponding anomalies.")
