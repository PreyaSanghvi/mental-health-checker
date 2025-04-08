import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Set up the Streamlit app
st.set_page_config(page_title="Mental Health Checker", layout="centered")
st.title("🧠 Mental Health Stress Detection")
st.write("Predict if a person is **Stressed** or **Not Stressed** based on their responses.")

# Input fields
age = st.slider("Age", 15, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
work_interfere = st.selectbox("Does your mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"])
no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Are care options available?", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Is there a wellness program?", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Does your employer encourage help-seeking?", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"])
leave = st.selectbox("Ease of taking mental health leave", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
mental_health_consequence = st.selectbox("Consequences of discussing mental health", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Consequences of discussing physical health", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Can you talk to coworkers about mental health?", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Can you talk to your supervisor?", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Willing to discuss mental health in interview?", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Willing to discuss physical health in interview?", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Should mental & physical health be treated the same?", ["Yes", "No", "Don't know"])
obs_consequence = st.selectbox("Seen negative consequences of mental health disclosure?", ["Yes", "No"])
comments_length = st.slider("Number of words in optional comments", 0, 500, 10)

# Encode inputs (Label Encoding as per your preprocessing)
mapping = {
    "Yes": 1, "No": 0, "Other": 2, "Male": 1, "Female": 0,
    "Don't know": 2, "Not sure": 2, "Maybe": 2, "Some of them": 2,
    "Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3,
    "Very easy": 0, "Somewhat easy": 1, "Don't know": 2,
    "Somewhat difficult": 3, "Very difficult": 4,
    "1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3,
    "500-1000": 4, "More than 1000": 5
}

input_list = [age,
              mapping.get(gender, 2),
              mapping[self_employed],
              mapping[family_history],
              mapping[work_interfere],
              mapping[no_employees],
              mapping[remote_work],
              mapping[tech_company],
              mapping[benefits],
              mapping[care_options],
              mapping[wellness_program],
              mapping[seek_help],
              mapping[anonymity],
              mapping[leave],
              mapping[mental_health_consequence],
              mapping[phys_health_consequence],
              mapping[coworkers],
              mapping[supervisor],
              mapping[mental_health_interview],
              mapping[phys_health_interview],
              mapping[mental_vs_physical],
              mapping[obs_consequence],
              comments_length
             ]

# Convert input to DataFrame
columns = ['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere', 
           'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
           'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
           'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
           'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments_length']

data = pd.DataFrame([input_list], columns=columns)

# Predict
# Predict
if st.button("Predict"):
    st.write("Input Data:")
    st.write(data)
    st.write("Any NaNs?", data.isnull().sum())

    prediction = model.predict(data)[0]
    result = "Stressed" if prediction == 1 else "Not Stressed"
    st.success(f"🧾 Prediction: The person is **{result}**.")

