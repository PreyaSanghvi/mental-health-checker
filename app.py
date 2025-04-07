import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('mental_health_model.pkl')

st.title("🧠 Mental Health Chatbot")

# User Inputs
name = st.text_input("What is your name?")
age = st.number_input("Your age:", min_value=10, max_value=100)
work_interfere = st.selectbox("Does mental health interfere with your work?", ['Never', 'Rarely', 'Sometimes', 'Often'])
benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No', "Don't know"])

# Add any other fields based on your dataset

if st.button("Submit"):
    data = pd.DataFrame([[age, work_interfere, benefits]], columns=['Age', 'work_interfere', 'benefits'])
    # Example encoding
data['work_interfere'] = data['work_interfere'].map({'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3})
data['benefits'] = data['benefits'].map({'Yes': 1, 'No': 0, "Don't know": 2})

    prediction = model.predict(data)[0]
    st.success(f"Hello {name}, based on your inputs, you are predicted as: **{prediction}**")
    
    # Save data to CSV (Optional)
    data['Prediction'] = prediction
    data.to_csv("new_user_data.csv", mode='a', header=False, index=False)
