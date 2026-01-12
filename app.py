import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model pipeline
# This includes the Scaler, Encoder, and Linear Regression math
model = joblib.load('salary_model.pkl')

st.set_page_config(page_title="Salary Predictor", page_icon="💰")

st.title("💰 Employee Salary Prediction")
st.markdown("### Powered by Azure Machine Learning")
st.info("This app uses a Linear Regression model trained on 6,000+ records to predict fair compensation.")

# 2. User Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])

with col2:
    job_title = st.text_input("Job Title", "Software Engineer")
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=5.0)

# 3. Prediction Logic
if st.button("Calculate Salary"):
    # Create a DataFrame matching the training schema
    input_data = pd.DataFrame([[age, gender, education, job_title, experience]], 
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"The predicted annual salary is: **${prediction[0]:,.2f}**")
