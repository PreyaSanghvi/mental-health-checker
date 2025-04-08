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

st.title("Mental Health Checker")

# Collect user inputs
with st.form("mental_health_form"):
    age = st.number_input("Age", min_value=1, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    occupation = st.selectbox("Occupation", ["Student", "Working Professional", "Unemployed", "Other"])
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, step=0.5)
    quality_sleep = st.slider("Quality of Sleep (1-5)", 1, 5)
    physical_activity = st.slider("Physical Activity Level (1-5)", 1, 5)
    stress_level = st.slider("Stress Level (1-10)", 1, 10)
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200)
    daily_steps = st.number_input("Daily Steps", min_value=0)
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
    screen_time = st.number_input("Daily Screen Time (hours)", min_value=0)
    work_life = st.selectbox("Work-Life Balance", ["Poor", "Average", "Good"])
    self_esteem = st.selectbox("Self Esteem", ["Low", "Moderate", "High"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    social_support = st.selectbox("Social Support", ["Low", "Moderate", "High"])
    emotional_state = st.selectbox("AI-Detected Emotional State", ["Calm", "Happy", "Sad"])
    
    submit = st.form_submit_button("Check My Mental Health")

if submit:
    # Prepare one-hot encoded emotional state
    emotional_state_dict = {
        'AI-Detected Emotional State_Calm': 1 if emotional_state == "Calm" else 0,
        'AI-Detected Emotional State_Happy': 1 if emotional_state == "Happy" else 0,
        'AI-Detected Emotional State_Sad': 1 if emotional_state == "Sad" else 0
    }

    # Combine all user input into a single dictionary
    input_data = {
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
        'Daily Screen Time': screen_time,
        'Work-Life Balance': work_life,
        'Self Esteem': self_esteem,
        'Family History of Mental Illness': family_history,
        'Social Support': social_support,
        **emotional_state_dict
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Align with expected model input
    input_df = input_df.reindex(columns=expected_columns)
    input_df = input_df.fillna(0)
# Reindex input_df with expected columns
input_df = input_df.reindex(columns=expected_columns)

# FILL any missing values just in case
input_df = input_df.fillna(0)

# Ensure all columns exist (add if missing due to form mismatch)
missing_cols = set(expected_columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Check column order again
input_df = input_df[expected_columns]

# Now prediction
prediction = model.predict(input_df)[0]

    st.success(f"Predicted Mental Health State: **{prediction}**")

