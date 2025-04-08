import streamlit as st
import joblib
import pandas as pd
from datetime import date

# Load trained model
model = joblib.load("mental_health_model.pkl")

# Expected features from model
expected_columns = [
    "Age", "Symptom Severity (1-10)", "Mood Score (1-10)", "Sleep Quality (1-10)",
    "Physical Activity (hrs/week)", "Treatment Start Date", "Treatment Duration (weeks)",
    "Stress Level (1-10)", "Treatment Progress (1-10)", "Adherence to Treatment (%)",
    "Gender_male", "Diagnosis_Generalized Anxiety", "Diagnosis_Major Depressive Disorder",
    "Diagnosis_Panic Disorder", "Medication_Antipsychotics", "Medication_Anxiolytics",
    "Medication_Benzodiazepines", "Medication_Mood Stabilizers", "Medication_SSRIs",
    "Therapy Type_Dialectical Behavioral Therapy", "Therapy Type_Interpersonal Therapy",
    "Therapy Type_Mindfulness-Based Therapy", "Outcome_Improved", "Outcome_No Change",
    "AI-Detected Emotional State_Depressed", "AI-Detected Emotional State_Excited",
    "AI-Detected Emotional State_Happy", "AI-Detected Emotional State_Neutral"
]

# Streamlit app
st.set_page_config(page_title="Mental Health Checker", layout="centered")
st.title("🧠 Mental Health Prediction")
st.markdown("Fill out the form below to receive insights about your mental health progress.")

with st.form("mental_health_form"):
    age = st.number_input("Age", 1, 100)
    severity = st.slider("Symptom Severity (1-10)", 1, 10)
    mood = st.slider("Mood Score (1-10)", 1, 10)
    sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10)
    physical_activity = st.number_input("Physical Activity (hrs/week)", 0.0, 168.0)
    start_date = st.date_input("Treatment Start Date")
    today = date.today()
    treatment_duration = (today - start_date).days // 7
    stress_level = st.slider("Stress Level (1-10)", 1, 10)
    treatment_progress = st.slider("Treatment Progress (1-10)", 1, 10)
    adherence = st.slider("Adherence to Treatment (%)", 0, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    diagnosis = st.multiselect("Diagnosis", ["Generalized Anxiety", "Major Depressive Disorder", "Panic Disorder"])
    medications = st.multiselect("Medications", ["Antipsychotics", "Anxiolytics", "Benzodiazepines", "Mood Stabilizers", "SSRIs"])
    therapy = st.selectbox("Therapy Type", ["Dialectical Behavioral Therapy", "Interpersonal Therapy", "Mindfulness-Based Therapy"])
    outcome = st.selectbox("Outcome", ["Improved", "No Change", "Worsened"])
    emotion = st.selectbox("AI-Detected Emotional State", ["Depressed", "Excited", "Happy", "Neutral"])
    submit = st.form_submit_button("Check My Mental Health")

if submit:
    # Base input dictionary
    input_data = {col: 0 for col in expected_columns}
    
    # Fill direct values
    input_data["Age"] = age
    input_data["Symptom Severity (1-10)"] = severity
    input_data["Mood Score (1-10)"] = mood
    input_data["Sleep Quality (1-10)"] = sleep_quality
    input_data["Physical Activity (hrs/week)"] = physical_activity
    input_data["Treatment Start Date"] = start_date.toordinal()
    input_data["Treatment Duration (weeks)"] = treatment_duration
    input_data["Stress Level (1-10)"] = stress_level
    input_data["Treatment Progress (1-10)"] = treatment_progress
    input_data["Adherence to Treatment (%)"] = adherence
    input_data["Gender_male"] = 1 if gender == "Male" else 0

    # Set diagnosis features
    for diag in diagnosis:
        key = f"Diagnosis_{diag}"
        if key in input_data:
            input_data[key] = 1

    # Set medications
    for med in medications:
        key = f"Medication_{med}"
        if key in input_data:
            input_data[key] = 1

    # Set therapy type
    therapy_key = f"Therapy Type_{therapy}"
    if therapy_key in input_data:
        input_data[therapy_key] = 1

    # Set outcome
    outcome_key = f"Outcome_{outcome}"
    if outcome_key in input_data:
        input_data[outcome_key] = 1

    # Set emotional state
    emotion_key = f"AI-Detected Emotional State_{emotion}"
    if emotion_key in input_data:
        input_data[emotion_key] = 1

    # Create DataFrame and predict
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.success(f"🧠 Predicted Mental Health Status: **{prediction}**")
