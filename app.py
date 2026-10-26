import streamlit as st
import joblib
import pandas as pd

model = joblib.load("diabetes_model.joblib")

st.title("Diabetes Risk Prediction")
st.write("Enter the patient's measurements.")

pregnancies = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Blood pressure (mm Hg)", 0, 200, 70)
skin = st.number_input("Skin thickness (mm)", 0, 100, 20)
insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
pedigree = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 33)

if st.button("Predict"):
    features = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin,
                              insulin, bmi, pedigree, age]],
                            columns=["Pregnancies", "Glucose", "BloodPressure",
                                     "SkinThickness", "Insulin", "BMI",
                                     "DiabetesPedigreeFunction", "Age"])
    result = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    if result == 1:
        st.error(f"Likely diabetic (probability {prob:.2f})")
    else:
        st.success(f"Likely not diabetic (probability {prob:.2f})")