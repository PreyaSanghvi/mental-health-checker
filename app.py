import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Define expected column order (same as during training)
expected_columns = ['Age' 'Symptom Severity (1-10)' 'Mood Score (1-10)'
 'Sleep Quality (1-10)' 'Physical Activity (hrs/week)'
 'Treatment Start Date' 'Treatment Duration (weeks)' 'Stress Level (1-10)'
 'Treatment Progress (1-10)' 'Adherence to Treatment (%)' 'Gender_male'
 'Diagnosis_Generalized Anxiety' 'Diagnosis_Major Depressive Disorder'
 'Diagnosis_Panic Disorder' 'Medication_Antipsychotics'
 'Medication_Anxiolytics' 'Medication_Benzodiazepines'
 'Medication_Mood Stabilizers' 'Medication_SSRIs'
 'Therapy Type_Dialectical Behavioral Therapy'
 'Therapy Type_Interpersonal Therapy'
 'Therapy Type_Mindfulness-Based Therapy' 'Outcome_Improved'
 'Outcome_No Change' 'AI-Detected Emotional State_Depressed'
 'AI-Detected Emotional State_Excited' 'AI-Detected Emotional State_Happy'
 'AI-Detected Emotional State_Neutral']

# Manual mappings used during training
gender_map = {"Male": 0, "Female": 1, "Other": 2}
occupation_map = {"Student": 0, "Working Professional": 1, "Unemployed": 2, "Other": 3}
bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
bp_map = {"Low": 0, "Normal": 1, "High": 2}
work_life_map = {"Poor": 0, "Average": 1, "Good": 2}
esteem_map = {"Low": 0, "Moderate": 1, "High": 2}
family_map = {"No": 0, "Yes": 1}
support_map = {"Low": 0, "Moderate": 1, "High": 2}

st.title("🧠 Mental Health Checker")

# Input form
with st.form("mental_health_form"):
    age = st.number_input("Age", min_value=1, max_value=100)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, step=0.5)
    quality_sleep = st.slider("Quality of Sleep (1-5)", 1, 5)
    physical_activity = st.slider("Physical Activity Level (1-5)", 1, 5)
    stress_level = st.slider("Stress Level (1-10)", 1, 10)
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200)
    daily_steps = st.number_input("Daily Steps", min_value=0)
    bmi_category = st.selectbox("BMI Category", list(bmi_map.keys()))
    blood_pressure = st.selectbox("Blood Pressure", list(bp_map.keys()))
    screen_time = st.number_input("Daily Screen Time (hours)", min_value=0)
    work_life = st.selectbox("Work-Life Balance", list(work_life_map.keys()))
    self_esteem = st.selectbox("Self Esteem", list(esteem_map.keys()))
    family_history = st.selectbox("Family History of Mental Illness", list(family_map.keys()))
    social_support = st.selectbox("Social Support", list(support_map.keys()))
    emotional_state = st.selectbox("AI-Detected Emotional State", ["Calm", "Happy", "Sad"])
    
    submit = st.form_submit_button("Check My Mental Health")

if submit:
    # One-hot encoding for emotional state
    emotional_state_dict = {
        'AI-Detected Emotional State_Calm': 1 if emotional_state == "Calm" else 0,
        'AI-Detected Emotional State_Happy': 1 if emotional_state == "Happy" else 0,
        'AI-Detected Emotional State_Sad': 1 if emotional_state == "Sad" else 0
    }

    # User input as dictionary
    input_data = {
        'Age': age,
        'Gender': gender_map[gender],
        'Occupation': occupation_map[occupation],
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'BMI Category': bmi_map[bmi_category],
        'Blood Pressure': bp_map[blood_pressure],
        'Daily Screen Time': screen_time,
        'Work-Life Balance': work_life_map[work_life],
        'Self Esteem': esteem_map[self_esteem],
        'Family History of Mental Illness': family_map[family_history],
        'Social Support': support_map[social_support],
        **emotional_state_dict
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct order and no missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Mental Health State: **{prediction}**")
    except Exception as e:
        st.error("⚠️ Prediction failed. Details:")
        st.error(str(e))
