
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('/content/random_forest_model.pkl')
scaler = joblib.load('/content/scaler.pkl')

# Define the web app interface
st.title('Diabetes Prediction App')

st.write('Enter the following details to predict diabetes:')

# Input fields for user interaction
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin Level', min_value=0, max_value=500, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)

# Button to trigger prediction
if st.button('Predict'):
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Preprocess the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display prediction result
    if prediction[0] == 0:
        st.write('Prediction: No Diabetes')
    else:
        st.write('Prediction: Diabetes')
