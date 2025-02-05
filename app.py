import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import os
from sklearn.ensemble import RandomForestClassifier

# Function to download the file from Google Drive
def download_file_from_google_drive(url, destination):
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Check if the request is successful
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    else:
        return False

# URL to the Google Drive file
file_url = "https://drive.google.com/uc?id=1WOL9TQLPZ-RRon8vc1sDtY7IKuYH0ydt"
model_file = "rf_model_cpu.pkl"

# Check if the file already exists, if not, download it
if not os.path.exists(model_file):
    success = download_file_from_google_drive(file_url, model_file)
    if success:
        st.success("Model file downloaded successfully!")
    else:
        st.error("Failed to download model file.")

# Load the trained model after downloading it
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Real sensor names and corresponding model feature names
sensor_names = {
    "Cycle": "Cycle", 
    "OpSet1": "OpSet1", "OpSet2": "OpSet2", "OpSet3": "OpSet3",
    "Primary Temperature Reading": "SensorMeasure1", 
    "Secondary Temperature Reading": "SensorMeasure2", 
    "Tertiary Temperature Reading": "SensorMeasure3", 
    "Quaternary Temperature Reading": "SensorMeasure4", 
    "Primary Pressure Reading": "SensorMeasure5", 
    "Secondary Pressure Reading": "SensorMeasure6", 
    "Tertiary Pressure Reading": "SensorMeasure7", 
    "Quaternary Pressure Reading": "SensorMeasure8", 
    "Primary Speed Reading": "SensorMeasure9", 
    "Secondary Speed Reading": "SensorMeasure10", 
    "Tertiary Speed Reading": "SensorMeasure11", 
    "Quaternary Speed Reading": "SensorMeasure12", 
    "Primary Vibration Reading": "SensorMeasure13", 
    "Secondary Vibration Reading": "SensorMeasure14", 
    "Primary Flow Reading": "SensorMeasure15", 
    "Secondary Flow Reading": "SensorMeasure16", 
    "Tertiary Flow Reading": "SensorMeasure17", 
    "Pressure Ratio": "SensorMeasure18", 
    "Efficiency Indicator": "SensorMeasure19", 
    "Power Setting": "SensorMeasure20", 
    "Fuel Flow Rate": "SensorMeasure21"
}

# Predefined values for GOOD, MODERATE, VERY BAD
predefined_values = {
    "GOOD": [64,20.0004,0.7007,100.0,491.19,606.79,1477.26,1234.25,9.35,13.61,332.51,2323.71,8709.48,1.07,43.86,313.57,2387.77,8050.58,9.1851,0.02,364,2324,100.0,24.6,14.6684],
    "MODERATE": [213,10.0018,0.25,100.0,489.05,604.4,1492.63,1306.34,10.52,15.47,397.07,2318.98,8778.54,1.26,45.37,373.56,2388.16,8141.38,8.571,0.03,369,2319,100.0,28.74,17.2585],
    "VERY BAD": [263,10.0077,0.2501,100.0,489.05,604.86,1507.7,1318.06,10.52,15.47,401.91,2319.43,8816.35,1.27,45.7,379.16,2388.61,8170.26,8.4897,0.03,372,2319,100.0,28.85,17.3519]
}

# Streamlit UI
st.title("Engine Health Predictor")

# Autofill buttons
for label in predefined_values:
    if st.button(f"Autofill for {label}"):
        for real_name, model_name in sensor_names.items():
            st.session_state[model_name] = predefined_values[label][list(sensor_names.keys()).index(real_name)]

# Text input for pasting comma-separated values
user_input = st.text_area("Paste comma-separated values", "")

# Apply button for user input
if st.button("Apply"):
    if user_input:
        input_values = list(map(float, user_input.split(',')))
        
        if len(input_values) == len(sensor_names):
            for real_name, model_name in sensor_names.items():
                st.session_state[model_name] = input_values[list(sensor_names.keys()).index(real_name)]
        else:
            st.error("Invalid input. Ensure the number of values matches the required fields.")

# User input fields with real sensor names displayed
input_data = []
for real_name, model_name in sensor_names.items():
    value = st.number_input(real_name, key=model_name, value=st.session_state.get(model_name, 0.0))
    input_data.append(value)

# Convert input_data to a pandas DataFrame with dtype float32
input_df = pd.DataFrame([input_data], columns=list(sensor_names.values()))
input_df = input_df.astype('float32')

# Submit button
if st.button("Submit"):
    prediction = model.predict(input_df)[0]  # Pass DataFrame to model
    result_map = {0: "GOOD", 1: "MODERATE", 2: "VERY BAD"}
    st.success(f"Predicted Condition: {result_map.get(prediction, 'Unknown')}")
