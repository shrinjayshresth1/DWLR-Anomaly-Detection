import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Title of the app
st.title("Anomaly Detection in DWLR Data")

# Section to upload the dataset
st.header("Upload Your DWLR Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset successfully uploaded and processed!")

    # Display the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Exclude non-numeric columns (like 'Date') from the correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])

    # Correlation Matrix
    st.header("Correlation Matrix")
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(plt)

    # Anomaly Detection Function
    '''def detect_anomalies(data):
        # Use only numeric columns for anomaly detection
        numeric_data = data[['Water_Level_m', 'Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']]
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        data['Isolation_Forest_Anomaly'] = iso_forest.fit_predict(numeric_data)
        data['Isolation_Forest_Anomaly'] = np.where(data['Isolation_Forest_Anomaly'] == -1, 1, 0)

        # Apply Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        data['LOF_Anomaly'] = lof.fit_predict(numeric_data)
        data['LOF_Anomaly'] = np.where(data['LOF_Anomaly'] == -1, 1, 0)

        # Apply One-Class SVM
        oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
        data['One_Class_SVM_Anomaly'] = oc_svm.fit_predict(numeric_data)
        data['One_Class_SVM_Anomaly'] = np.where(data['One_Class_SVM_Anomaly'] == -1, 1, 0)

        return data'''
    
    from sklearn.impute import SimpleImputer

    def detect_anomalies(data):
       # Use only numeric columns for anomaly detection
       numeric_data = data[['Water_Level_m', 'Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']]
    
       # Impute missing values using SimpleImputer
       imputer = SimpleImputer(strategy='mean')
       numeric_data = imputer.fit_transform(numeric_data)
    
       # Apply Isolation Forest
       iso_forest = IsolationForest(contamination=0.05, random_state=42)
       data['Isolation_Forest_Anomaly'] = iso_forest.fit_predict(numeric_data)
       data['Isolation_Forest_Anomaly'] = np.where(data['Isolation_Forest_Anomaly'] == -1, 1, 0)

       # Apply Local Outlier Factor
       lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
       data['LOF_Anomaly'] = lof.fit_predict(numeric_data)
       data['LOF_Anomaly'] = np.where(data['LOF_Anomaly'] == -1, 1, 0)

      # Apply One-Class SVM
       oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
       data['One_Class_SVM_Anomaly'] = oc_svm.fit_predict(numeric_data)
       data['One_Class_SVM_Anomaly'] = np.where(data['One_Class_SVM_Anomaly'] == -1, 1, 0)

       return data

    # Detect Anomalies
    data = detect_anomalies(data)

    # Plot Anomalies
    def plot_anomalies(feature, title, anomaly_column):
        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data[feature], label=title, color='blue')
        plt.scatter(data[data[anomaly_column] == 1]['Date'], data[data[anomaly_column] == 1][feature],
                    color='red', label='Anomalies', marker='o')
        plt.title(f'{title} Over Time with Anomalies Highlighted')
        plt.xlabel('Date')
        plt.ylabel(title)
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    # Display results
    st.header("Anomaly Detection Results")
    for feature in ['Water_Level_m', 'Temperature_C', 'Rainfall_mm', 'pH', 'Dissolved_Oxygen_mg_L']:
        plot_anomalies(feature, feature, 'One_Class_SVM_Anomaly')

    # Summary of detected anomalies
    st.sidebar.header("Anomaly Detection Summary")
    st.sidebar.write(f"Detected {data['One_Class_SVM_Anomaly'].sum()} anomalies using One-Class SVM.")
else:
    st.warning("Please upload a CSV file to proceed.")
