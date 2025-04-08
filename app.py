import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Define expected column order
expected_columns = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps',
    'BMI Category', 'Blood Pressure', 'Daily Screen Time', 'Work-Life Balance',
    'Self Esteem', 'Family History of Mental Illness', 'Social Support',
    'AI-Detected Emotional State_Calm', 'AI-Detected Emotional State_Happy',
    'AI-Detected Emotional State_Sad'
]

# Manual label encoders (used during training)
gender_map = {"Male": 0, "Female": 1, "Other": 2}
occupation_map = {"Student": 0, "Working Professional": 1, "Unemployed": 2, "Other": 3}
bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
bp_map = {"Low": 0, "Normal": 1, "High": 2}
work_life_map = {"Poor": 0, "Average": 1, "Good": 2}
esteem_map = {"Low": 0, "Moderate": 1, "High": 2}
family_map = {"No": 0, "Yes": 1}
support_map = {"Low": 0, "Moderate": 1, "High": 2}

st.title("Mental Health Checker")

# Collect user inputs
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
    # One-hot encode emotional state
    emotional_state_dict = {
        'AI-Detected Emotional State_Calm': 1 if emotional_state == "Calm" else 0,
        'AI-Detected Emotional State_Happy': 1 if emotional_state == "Happy" else 0,
        'AI-Detected Emotional State_Sad': 1 if emotional_state == "Sad" else 0
    }

    # Prepare the input dictionary
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

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Align columns and fill missing
    input_df = input_df.reindex(columns=expected_columns)
    input_df = input_df.fillna(0)

    # Final safety check for column order
    missing_cols = set(expected_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[expected_columns]

    # Prediction
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Mental Health State: **{prediction}**")
