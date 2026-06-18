import streamlit as st
import pandas as pd
import xgboost as xgb

st.title("Student Performance Prediction App")

# Input fields
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
assignment = st.number_input("Assignment Score", min_value=0, max_value=100)
internal = st.number_input("Internal Assessment", min_value=0, max_value=100)
study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=50)

# Predict button
if st.button("Predict Result"):
    # Create dataframe for model input
    input_data = pd.DataFrame({
        "Attendance":[attendance],
        "Assignment":[assignment],
        "InternalAssessment":[internal],
        "StudyHours":[study_hours]
    })
    
    # Load trained model
    model = xgb.XGBClassifier()
    model.load_model("student_model.json")   # <-- apna trained model yaha rakho
    
    # Prediction
    prediction = model.predict(input_data)
    result = "Pass ✅" if prediction[0]==1 else "Fail ❌"
    
    st.success(f"Prediction: {result}")
