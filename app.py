import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

st.title("Diabetes Prediction System")

st.write("Enter patient details")

# Input fields
pregnancies = st.number_input("Pregnancies",min_value=0)
glucose = st.number_input("Glucose",min_value=0)
bloodpressure = st.number_input("Blood Pressure",min_value=0)
skinthickness = st.number_input("Skin Thickness",min_value=0)
insulin = st.number_input("Insulin",min_value=0)
bmi = st.number_input("BMI",min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function",min_value=0.0)
age = st.number_input("Age",min_value=0)

# Prediction button
if st.button("Predict"):

    sample = pd.DataFrame(
        [[pregnancies,
          glucose,
          bloodpressure,
          skinthickness,
          insulin,
          bmi,
          dpf,
          age]],
        columns=[
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
    )

    # Scaling
    sample_scaled = scaler.transform(sample)

    # PCA
    sample_pca = pca.transform(sample_scaled)

    # Prediction
    prediction = model.predict(sample_pca)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Healthy")