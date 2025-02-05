import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model (Ensure it's a CPU-based model saved as .pkl)
with open('rf_model_cpu.pkl', 'rb') as file:
    model = pickle.load(file)

# Updated fields based on the exact feature names from the model
fields = [
    "Cycle", 
    "OpSet1", "OpSet2", "OpSet3",
    "SensorMeasure1", "SensorMeasure2", "SensorMeasure3", "SensorMeasure4", 
    "SensorMeasure5", "SensorMeasure6", "SensorMeasure7", "SensorMeasure8", 
    "SensorMeasure9", "SensorMeasure10", "SensorMeasure11", "SensorMeasure12", 
    "SensorMeasure13", "SensorMeasure14", 
    "SensorMeasure15", "SensorMeasure16", "SensorMeasure17", 
    "SensorMeasure18", "SensorMeasure19", "SensorMeasure20", "SensorMeasure21"
]

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
        for i, field in enumerate(fields):
            st.session_state[field] = predefined_values[label][i]

# Text input for pasting comma-separated values
user_input = st.text_area("Paste comma-separated values", "")

# Apply button for user input
if st.button("Apply"):
    if user_input:
        input_values = list(map(float, user_input.split(',')))
        
        if len(input_values) == len(fields):
            for i, field in enumerate(fields):
                st.session_state[field] = input_values[i]
        else:
            st.error("Invalid input. Ensure the number of values matches the required fields.")

# User input fields
input_data = []
for field in fields:
    value = st.number_input(field, key=field, value=st.session_state.get(field, 0.0))
    input_data.append(value)

# Convert input_data to a pandas DataFrame with dtype float32
input_df = pd.DataFrame([input_data], columns=fields)
input_df = input_df.astype('float32')

# Submit button
if st.button("Submit"):
    prediction = model.predict(input_df)[0]  # Pass DataFrame to model
    result_map = {0: "GOOD", 1: "MODERATE", 2: "VERY BAD"}
    st.success(f"Predicted Condition: {result_map.get(prediction, 'Unknown')}")
