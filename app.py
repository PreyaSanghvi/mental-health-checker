import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mental_health_model.pkl")

st.title("🧠 Mental Health Chatbot")

# Input
name = st.text_input("What is your name?")
age = st.number_input("Your age:", min_value=10, max_value=100)
work_interfere = st.selectbox("Do you feel your mental health interferes with your work?", ['Never', 'Rarely', 'Sometimes', 'Often'])
benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No', "Don't know"])

# Manual encoding (based on your training preprocessing)
work_interfere_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
benefits_map = {'Yes': 1, 'No': 0, "Don't know": 2}

# Transform input to numerical
work_interfere_val = work_interfere_map[work_interfere]
benefits_val = benefits_map[benefits]

# Add more preprocessing if needed

if st.button("Submit"):
    data = pd.DataFrame([[age, work_interfere_val, benefits_val]], 
                        columns=['Age', 'work_interfere', 'benefits'])

    prediction = model.predict(data)[0]
    
    st.write(f"Hello {name}, based on your responses, our system predicts: **{prediction}**")

    # Append data to CSV
    data['Prediction'] = prediction
    data.to_csv("new_user_data.csv", mode='a', header=False, index=False)

