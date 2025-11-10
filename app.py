import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load('student_score_model.pkl')

# Page setup
st.set_page_config(page_title="Student Score Predictor", page_icon="📚", layout="centered")

st.title("📚 Student Score Predictor")
st.write("Predict a student’s score based on hours studied using a simple Linear Regression model.")

# Load dataset
df = pd.read_csv("students_score.csv")

# Show dataset preview
with st.expander("📄 View Dataset"):
    st.dataframe(df)

# Input hours from user
hours = st.number_input("Enter study hours:", min_value=0.0, max_value=24.0, step=0.25)

# Predict button
if st.button("🔮 Predict Score"):
    predicted_score = model.predict(np.array([[hours]]))
    predicted_score = np.clip(predicted_score, 0, 100)  # Limit between 0 and 100
    st.success(f"🎯 Predicted Score: **{predicted_score[0]:.2f}%**")


    # Plot regression line
    st.subheader("📊 Study Hours vs Scores")
    fig, ax = plt.subplots()
    ax.scatter(df['Hours'], df['Scores'], color='blue', label='Actual Scores')
    ax.plot(df['Hours'], model.predict(df[['Hours']]), color='red', label='Regression Line')
    ax.scatter(hours, predicted_score, color='green', s=100, label='Your Prediction')
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Scores")
    ax.legend()
    st.pyplot(fig)

st.caption("Made using Streamlit & Scikit-Learn")
