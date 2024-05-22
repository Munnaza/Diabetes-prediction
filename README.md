# Diabetes-prediction
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
file_path = '/content/diabetes.xlsx'  # Update the path if necessary
diabetes = pd.read_excel(file_path)
# Data Cleaning
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_with_zeros:
    diabetes[column].replace(0, np.nan, inplace=True)
    diabetes[column].fillna(diabetes[column].median(), inplace=True)

# Data Preparation
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to balance the classes in the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Save the model and scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
# Define new data for prediction
new_data = pd.DataFrame({
    'Pregnancies': [6],             # E
    'Glucose': [148],               # Example value, replace with actual values
    'BloodPressure': [72],          # Example value, replace with actual values
    'SkinThickness': [35],          # Example value, replace with actual values
    'Insulin': [0],                 # Example value, replace with actual values
    'BMI': [33.6],                  # Example value, replace with actual values
    'DiabetesPedigreeFunction': [0.627],  # Example value, replace with actual values
    'Age': [50]                     # Example value, replace with actual values
})

# Make predictions on the new data
predictions = rf_model.predict(new_data)

# Display the predictions
print("Predictions:")
print(predictions)
import joblib
import pandas as pd

# Load the trained model
rf_model = joblib.load('random_forest_model.pkl')  # Load your serialized model file

def make_prediction(input_data):
    # Preprocess the input data
    # Ensure that input_data is preprocessed in the same way as your training data

    # Make predictions using the loaded model
    predictions = rf_model.predict(input_data)

    # Return the predictions
    return predictions

# Example usage
input_data = pd.DataFrame({
    'Pregnancies': [6],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50]
})

predicted_outcome = make_prediction(input_data)
print("Predicted Outcome:", predicted_outcome)
model = joblib.load('random_forest_model.pkl')
import joblib

# Assuming `model` is your trained Random Forest model
model = RandomForestClassifier()
# Train your model...

# Save the trained model to a file
joblib.dump(model, 'random_forest_model.pkl')
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('/content/random_forest_model.pkl')

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

    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction result
    if prediction[0] == 0:
        st.write('Prediction: No Diabetes')
    else:
        st.write('Prediction: Diabetes')
        # Write the code to a file named app.py
with open('app.py', 'w') as file:
    file.write(app_code)
!streamlit run .py

