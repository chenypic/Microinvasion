
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('xgboost_model.pkl')

# Define feature options
cp_options = {
    1: 'Typical angina (1)',
    2: 'Atypical angina (2)',
    3: 'Non-anginal pain (3)',
    4: 'Asymptomatic (4)'
}

restecg_options = {
    0: 'Normal (0)',
    1: 'ST-T wave abnormality (1)',
    2: 'Left ventricular hypertrophy (2)'
}

slope_options = {
    1: 'Upsloping (1)',
    2: 'Flat (2)',
    3: 'Downsloping (3)'
}

thal_options = {
    1: 'Normal (1)',
    2: 'Fixed defect (2)',
    3: 'Reversible defect (3)'
}

# Define feature names
feature_names = [
    "age", "HPV", "TCT_HSIL", "ECC", "margin","Transformation"
]

# Streamlit user interface
st.title("Microinvasion Prediction System")

# age: numerical input
#age = st.number_input("Age:", min_value=1, max_value=120, value=50)

age = st.selectbox("Age (0=<48, 1=>48):", options=[0, 1], format_func=lambda x: '<48 (0)' if x == 0 else '>48 (1)')

# sex: categorical selection
HPV = st.selectbox("HPV (0=Negative, 1=Positive):", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

# cp: categorical selection
TCT_HSIL = st.selectbox("TCT_HSIL (0=Negative, 1=Positive):", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

# trestbps: numerical input
ECC = st.selectbox("ECC (0=Negative, 1=Positive):", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

# chol: numerical input
margin = st.selectbox("margin (0=Negative, 1=Positive):", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

# fbs: categorical selection
Transformation = st.selectbox("Transformation (0=Negative, 1=Positive):", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')


# Process inputs and make predictions
feature_values = [age, HPV, TCT_HSIL, ECC, margin, Transformation]
features = np.array([feature_values])


if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Microinvasion . "
            f"The model predicts that your probability of having Microinvasion  is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Microinvasion . "
            f"The model predicts that your probability of not having Microinvasion  is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)

    st.image("shap_force_plot.png")